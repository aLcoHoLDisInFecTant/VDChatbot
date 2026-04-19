"""
Batch evaluation for the LangGraph medical triage system (Agentic RAG).

Metrics:
  - Routing Accuracy: compare executed node path + tool usage to category expectations.
  - Factual Accuracy / Hallucination: LLM-as-judge vs. expected facts (1 = factual, 0 = hallucination).

Prerequisites:
  - Same environment as the app (e.g. HKBU_API_KEY when Config.LLM_TYPE == "hkbu").
  - `utils.tools_config.get_tools()` must register both `retrieve` and `multiply` for Utility routing to match.
  - ChromaDB on disk; optional Postgres for graph store (falls back to in-memory).

Usage:
  python evaluate_system.py
  python evaluate_system.py --csv my_results.csv
  python evaluate_system.py --skip-judge   # routing + latency only (no LLM judge calls)
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import logging
import math
import re
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from ragAgent import (
    ToolConfig,
    create_graph,
    ConnectionPool,
    ConnectionPoolError,
    monitor_connection_pool,
)
from utils.config import Config
from utils.llms import get_llm
from utils.tools_config import get_tools
from utils import pdfSplitTest_Ch
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

Category = Literal["general", "utility", "vertical"]


def build_test_dataset() -> List[Dict[str, Any]]:
    """50 English-only test cases: 15 general, 15 utility, 20 vertical."""
    rows: List[Dict[str, Any]] = []

    general = [
        ("Hello, how are you?", "Assistant should greet politely; no medical claims or tool use implied."),
        ("Hi there!", "Brief friendly reply; no patient-specific medical facts."),
        ("Good morning.", "Polite acknowledgment; no fabricated health data."),
        ("Who am I?", "May say it cannot know identity without context; no invented personal medical history."),
        ("How is the weather today?", "Should not invent live weather; may explain it cannot access real-time data."),
        ("Thank you for your help.", "Brief polite closure; no medical advice required."),
        ("What can you do?", "Describe capabilities honestly; do not claim impossible real-time data access."),
        ("Tell me a short joke.", "Light humor allowed; no medical misinformation."),
        ("Goodbye.", "Polite farewell."),
        ("How are you today?", "Conversational reply; no diagnostic statements."),
        ("What is your name?", "May give a role name; no patient PHI."),
        ("Nice to meet you.", "Polite reciprocation."),
        ("See you later.", "Short polite sign-off."),
        ("Please describe yourself briefly.", "Honest capability summary; English interaction."),
        ("What day is it?", "May note it cannot know the real calendar without tools; do not invent a false date as fact."),
    ]
    for i, (q, facts) in enumerate(general, start=1):
        rows.append({"id": f"G{i:02d}", "category": "general", "query": q, "expected_facts": facts})

    utility = [
        ("What is 3 times 4?", "The correct product is 12."),
        ("Calculate 15 * 8.", "The correct product is 120."),
        ("Multiply 7 and 9.", "The correct product is 63."),
        ("What is 12 multiplied by 5?", "The correct product is 60."),
        ("What is 6 times 3?", "The correct product is 18."),
        ("Calculate 25 times 4.", "The correct product is 100."),
        ("What is 11 * 11?", "The correct product is 121."),
        ("Multiply 0.5 and 8.", "The correct product is 4."),
        ("What is 99 times 1?", "The correct product is 99."),
        ("Calculate 13 * 7 for me.", "The correct product is 91."),
        ("What is 20 times 15?", "The correct product is 300."),
        ("Multiply 2.5 by 4.", "The correct product is 10."),
        ("What is 8 times 125?", "The correct product is 1000."),
        ("Calculate 17 * 3.", "The correct product is 51."),
        ("What is 14 times 6?", "The correct product is 84."),
    ]
    for i, (q, facts) in enumerate(utility, start=1):
        rows.append({"id": f"U{i:02d}", "category": "utility", "query": q, "expected_facts": facts})

    vertical = [
        ("张三九的基本信息是什么（性别、年龄、职业、地址、联系方式、紧急联系人）？", "Answer must rely on retrieved passages from 健康档案.pdf; no invented identity details."),
        ("列出张三九的既往病史和确诊年份。", "Only list conditions and timelines supported by retrieved text; otherwise say unknown."),
        ("张三九有哪些过敏史？分别有什么反应？", "Allergy items and reactions must come from retrieval; do not fabricate."),
        ("张三九的家族病史是什么（父亲/母亲分别有哪些疾病史）？", "Family history must match retrieved text; do not add new diseases."),
        ("总结张三九的生活方式与习惯（饮食、运动、吸烟、饮酒、睡眠）。", "Summarize only what retrieval contains; no added habits."),
        ("张三九最近的生命体征有哪些（身高、体重、BMI、血压、心率）？如果记录里没有就说明未找到。", "Numeric values must match retrieved evidence; avoid unsupported numbers."),
        ("张三九的血糖和血脂化验结果分别是多少？哪些指标异常？", "Only report values and interpretations supported by retrieval."),
        ("张三九有哪些影像或检查结果（心电图、胸片、腹部超声等）？请概括结论。", "Summarize imaging/exam findings only if present in retrieved text."),
        ("张三九有哪些治疗或手术记录？分别发生在什么时间？", "Procedures and dates must match retrieved evidence."),
        ("张三九的随访计划或医生建议有哪些？", "Recommendations must be grounded in retrieved content."),
        ("李四六的基本信息是什么？她的紧急联系人是谁？", "Use retrieved evidence only; do not guess."),
        ("列出李四六的既往病史（如有妊娠相关情况也包含）以及对应年份。", "Must match retrieved record entries; do not invent."),
        ("李四六有哪些过敏史（药物/食物）？分别有什么反应？", "Only from retrieval; if missing, say not found."),
        ("总结李四六的生活方式与风险因素。", "Use retrieved evidence only."),
        ("李四六的血糖/血脂化验结果是什么？哪些异常？", "Only report values supported by retrieval."),
        ("王五的基本信息是什么（年龄、职业、婚姻状况）？", "Only answer if supported by retrieved passages."),
        ("对比张三九和李四六：谁的糖尿病风险更高？依据档案说明原因。", "Comparison must cite retrieved facts; avoid new lab values."),
        ("对比张三九和李四六：谁的心血管风险更高？依据档案说明原因。", "Comparison must cite retrieved facts; no invented numbers."),
        ("档案里是否提到高血压的管理或用药？请总结记录到的内容。", "Medication names/management details must appear in retrieved text."),
        ("健康档案里有哪些信息缺失或无法找到？请列出未找到的字段。", "If information is absent, explicitly say it is not found in the retrieved records."),
    ]
    for i, (q, facts) in enumerate(vertical, start=1):
        rows.append({"id": f"V{i:02d}", "category": "vertical", "query": q, "expected_facts": facts})

    assert len(rows) == 50
    return rows


def _collect_tool_calls_from_message(msg: BaseMessage) -> List[str]:
    names: List[str] = []
    if not isinstance(msg, AIMessage):
        return names
    tcs = getattr(msg, "tool_calls", None) or []
    for tc in tcs:
        if isinstance(tc, dict) and tc.get("name"):
            names.append(str(tc["name"]))
        elif hasattr(tc, "name"):
            names.append(str(tc.name))
    return names


def _extract_final_assistant_text(messages: Sequence[BaseMessage]) -> str:
    """Last non-tool-call AI message content (similar to graph_response)."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and (getattr(msg, "tool_calls", None) or []):
            continue
        content = getattr(msg, "content", None)
        if content is None:
            continue
        text = content if isinstance(content, str) else str(content)
        if text.strip():
            return text.strip()
    return ""


def run_graph_once(
    graph,
    user_input: str,
    config: dict,
) -> Tuple[List[str], List[str], str, float, List[str]]:
    """
    Returns: node_path (ordered), tools_called (ordered, may repeat), final_output, latency_sec, retrieved_texts.

    Uses stream_mode="updates" (LangGraph 1.x) so each chunk keys are real node names, not state fields.
    """
    t0 = time.perf_counter()
    path: List[str] = []
    tools_called: List[str] = []
    last_messages: List[BaseMessage] = []
    retrieved_texts: List[str] = []

    inputs = {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}

    # LangGraph 1.x defaults to stream_mode="values" (full state per step). Use "updates"
    # so each chunk is {node_name: partial_state} and node paths match real graph nodes.
    cfg = dict(config) if config else {}
    if "recursion_limit" not in cfg:
        cfg = {**cfg, "recursion_limit": 50}

    for event in graph.stream(inputs, cfg, stream_mode="updates"):
        if not isinstance(event, dict):
            continue
        for node_name, payload in event.items():
            if node_name.startswith("__"):
                continue
            path.append(node_name)

            if isinstance(payload, dict) and "messages" in payload:
                msgs = payload["messages"]
                if isinstance(msgs, list) and msgs:
                    last_messages = msgs
                    last = msgs[-1]
                    tools_called.extend(_collect_tool_calls_from_message(last))
                    for m in msgs:
                        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "retrieve":
                            content = getattr(m, "content", None)
                            if content is None:
                                continue
                            text = content if isinstance(content, str) else str(content)
                            text = text.strip()
                            if text:
                                retrieved_texts.append(text)

    latency_sec = time.perf_counter() - t0
    final_text = _extract_final_assistant_text(last_messages) if last_messages else ""
    return path, tools_called, final_text, latency_sec, retrieved_texts


def routing_correct(category: Category, path: Sequence[str], tools_called: Sequence[str]) -> bool:
    """
    Expected routing (aligned with ragAgent):
    - general: no call_tools (small talk ends at agent -> END).
    - utility: call_tools, multiply used, no grade_documents (non-retrieval tools route to generate).
    - vertical: call_tools, retrieve used, grade_documents, generate.
    """
    p = list(path)
    tc = list(tools_called)

    if category == "general":
        return "call_tools" not in p

    if category == "utility":
        return "call_tools" in p and "multiply" in tc and "grade_documents" not in p

    if category == "vertical":
        return (
            "call_tools" in p
            and "retrieve" in tc
            and "grade_documents" in p
            and "generate" in p
        )

    return False


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("nan")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _split_sentences(text: str, *, max_sents: int = 25) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    t = re.sub(r"\s+", " ", t)
    parts = re.split(r"(?<=[.!?])\s+", t)
    out = [p.strip() for p in parts if p and p.strip()]
    return out[:max_sents]


def _split_evidence(text: str, *, max_chunks: int = 16) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    chunks = [c.strip() for c in t.split("\n\n") if c and c.strip()]
    if len(chunks) < 2:
        chunks = [c.strip() for c in re.split(r"[\r\n]+", t) if c and c.strip()]
    return chunks[:max_chunks]


def _extract_numbers(text: str) -> List[str]:
    t = text or ""
    nums = re.findall(r"(?<![\w])[-+]?\d*\.?\d+(?![\w])", t)
    return [n for n in nums if n and n not in ("-", "+", ".")]


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _is_mostly_english(text: str) -> bool:
    t = text or ""
    letters = re.findall(r"[A-Za-z]", t)
    cjk = re.findall(r"[\u4e00-\u9fff]", t)
    return len(letters) >= 20 and len(cjk) <= 2


def _translate_to_chinese(llm_chat, text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    prompt = (
        "Translate the following text into Simplified Chinese. "
        "Only output the translation.\n\n"
        f"Text:\n{t}"
    )
    try:
        msg = llm_chat.invoke(prompt)
        out = getattr(msg, "content", "") if msg is not None else ""
        out = out if isinstance(out, str) else str(out)
        return out.strip() or t
    except Exception:
        return t


def ensure_chroma_seed(llm_embedding) -> None:
    health_pdf = "input/健康档案.pdf"

    vectorstore = Chroma(
        persist_directory=Config.CHROMADB_DIRECTORY,
        collection_name=Config.CHROMADB_COLLECTION_NAME,
        embedding_function=llm_embedding,
    )

    def _add_texts(texts: List[str], source: str) -> None:
        if not texts:
            return
        ids: List[str] = []
        metas: List[Dict[str, Any]] = []
        for i, tx in enumerate(texts):
            s = (tx or "").strip()
            if not s:
                continue
            h = hashlib.md5(s.encode("utf-8")).hexdigest()
            ids.append(f"{source}:{i}:{h}")
            metas.append({"source": source})
        if ids:
            vectorstore.add_texts(texts=[t.strip() for t in texts if t and t.strip()], ids=ids, metadatas=metas)

    try:
        health_chunks = pdfSplitTest_Ch.getParagraphs(health_pdf, page_numbers=None, min_line_length=1)
        _add_texts(list(health_chunks), "健康档案.pdf")
        logger.info("Chroma seeded: 健康档案.pdf (%d chunks)", len(health_chunks))
    except Exception as e:
        logger.warning("Failed to seed 健康档案.pdf into Chroma: %s", str(e)[:200])


def score_utility(expected_facts: str, actual_output: str) -> Tuple[int, str, Dict[str, Any]]:
    exp = expected_facts or ""
    out = (actual_output or "").strip()
    m = re.search(r"correct product is\s+([-+]?\d*\.?\d+)", exp, flags=re.IGNORECASE)
    if not m:
        return 0, "expected_parse_failed", {"expected_facts": exp[:200]}
    expected_num = m.group(1)
    nums = _extract_numbers(out)
    if not nums:
        return 0, f"no_number_found_expected_{expected_num}", {"expected": expected_num}
    got = nums[-1]
    try:
        ok = 1 if abs(float(got) - float(expected_num)) <= 1e-9 else 0
    except Exception:
        ok = 1 if got == expected_num else 0
    return ok, (f"expected_{expected_num}_got_{got}"), {"expected": expected_num, "got": got}


def score_general(actual_output: str) -> Tuple[int, str, Dict[str, Any]]:
    out = (actual_output or "").strip()
    if not out:
        return 0, "empty_output", {}
    suspicious_names = [
        "John Doe",
        "Kevin",
        "Jane Smith",
        "Anna",
        "Michael Brown",
        "Sarah Johnson",
        "David Lee",
        "Emma Wilson",
        "Robert Taylor",
    ]
    medical_markers = [
        "diagnosis",
        "hypertension",
        "diabetes",
        "cholesterol",
        "blood pressure",
        "mmhg",
        "mg/dl",
        "allergy",
        "medication",
        "lab result",
        "vitals",
        "imaging",
    ]
    has_name = any(n.lower() in out.lower() for n in suspicious_names)
    has_med = any(k in out.lower() for k in medical_markers)
    has_nums = bool(_extract_numbers(out))
    if has_name and (has_med or has_nums):
        return 0, "general_contains_patient_medical_facts", {"has_name": True, "has_med": has_med, "has_nums": has_nums}
    return 1, "general_ok", {}


def score_vertical(
    *,
    retrieved_texts: Sequence[str],
    actual_output: str,
    llm_embedding,
    llm_chat,
    sim_threshold: float,
    max_unsupported_ratio: float,
) -> Tuple[int, str, Dict[str, Any]]:
    out = (actual_output or "").strip()
    if out:
        m = re.search(r"\n\s*evidence\s*:?\s*\n", out, flags=re.IGNORECASE)
        if m:
            out = out[: m.start()].strip()
    evidence = "\n\n".join([t for t in retrieved_texts if t and t.strip()]).strip()
    if not out:
        return 0, "empty_output", {"evidence_present": bool(evidence)}

    if "vector store is unavailable in this runtime" in evidence.lower():
        return -1, "VECTORSTORE_UNAVAILABLE", {"evidence_present": True}

    if (not evidence) or ("no health-record passage" in evidence.lower()):
        deny_markers = [
            "not found",
            "no record",
            "cannot find",
            "couldn't find",
            "don’t have",
            "don't have",
            "missing",
            "unavailable",
            "unknown",
        ]
        ok_if_denies = any(m in out.lower() for m in deny_markers)
        has_nums = bool(_extract_numbers(out))
        if ok_if_denies and not has_nums:
            return 1, "no_evidence_acknowledged", {"evidence_present": False}
        return 0, "no_evidence_but_asserts_details", {"evidence_present": False, "has_nums": has_nums}

    deny_markers = [
        "not found",
        "no record",
        "cannot find",
        "couldn't find",
        "don’t have",
        "don't have",
        "missing",
        "unavailable",
        "unknown",
        "未找到",
        "找不到",
        "没有记录",
        "无法找到",
        "未知",
    ]

    chunks = _split_evidence(evidence)
    if not chunks:
        return 0, "insufficient_text_for_similarity", {"sentences": 0, "chunks": len(chunks)}

    def _filter_sents(ss: Sequence[str]) -> List[str]:
        out_s: List[str] = []
        for s in ss:
            st = (s or "").strip()
            if not st:
                continue
            low = st.lower()
            has_deny = any(m in low for m in deny_markers)
            has_nums = bool(_extract_numbers(st))
            if has_deny and not has_nums:
                continue
            out_s.append(st)
        return out_s

    def _score_with_sentences(sentences: List[str]) -> Tuple[int, str, Dict[str, Any]]:
        sents = _filter_sents(sentences)
        if not sents:
            return 0, "insufficient_text_for_similarity", {"sentences": 0, "chunks": len(chunks)}
        try:
            sent_vecs = llm_embedding.embed_documents(sents)
            chunk_vecs = llm_embedding.embed_documents(chunks)
        except Exception as e:
            return 0, f"embedding_error_{type(e).__name__}", {"error": str(e)[:200]}

        max_sims: List[float] = []
        for sv in sent_vecs:
            best = float("-inf")
            for cv in chunk_vecs:
                sim = _cosine_similarity(sv, cv)
                if math.isnan(sim):
                    continue
                if sim > best:
                    best = sim
            if best == float("-inf"):
                best = float("nan")
            max_sims.append(best)

        valid_sims = [x for x in max_sims if isinstance(x, float) and not math.isnan(x)]
        if not valid_sims:
            return 0, "similarity_all_nan", {"sentences": len(sents), "chunks": len(chunks)}

        min_max_sim = min(valid_sims)
        avg_max_sim = sum(valid_sims) / len(valid_sims)
        unsupported = [x for x in valid_sims if x < sim_threshold]
        unsupported_ratio = len(unsupported) / len(valid_sims) if valid_sims else 1.0

        nums = _extract_numbers(out)
        numeric_unverified = 0
        if nums:
            ev_low = evidence.lower()
            for n in nums:
                if n.lower() not in ev_low:
                    numeric_unverified += 1

        numeric_flag = bool(nums) and (numeric_unverified >= 2)
        ok = (unsupported_ratio <= max_unsupported_ratio) and (min_max_sim >= sim_threshold) and (not numeric_flag)
        reason = "supported_by_evidence" if ok else "not_supported_by_evidence"
        diag = {
            "min_max_sim": min_max_sim,
            "avg_max_sim": avg_max_sim,
            "unsupported_ratio": unsupported_ratio,
            "sim_threshold": sim_threshold,
            "max_unsupported_ratio": max_unsupported_ratio,
            "numbers_in_answer": len(nums),
            "numeric_unverified": numeric_unverified,
            "numeric_flag": numeric_flag,
        }
        return (1 if ok else 0), reason, diag

    if _has_cjk(evidence) and _is_mostly_english(out):
        translated = _translate_to_chinese(llm_chat, out)
        score_a = _score_with_sentences(_split_sentences(out))
        score_b = _score_with_sentences(_split_sentences(translated))
        if score_a[0] >= score_b[0]:
            chosen = score_a if score_a[0] == 1 else score_b
        else:
            chosen = score_b
        return chosen

    return _score_with_sentences(_split_sentences(out))

    


def _fmt_pct(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.2%}"


def print_markdown_report(stats: Dict[str, Any]) -> None:
    lines = [
        "",
        "## Evaluation Summary",
        "",
        "| Scope | Count | Routing Acc | Factual Acc | Avg Latency (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat in ("general", "utility", "vertical"):
        c = stats["by_category"][cat]
        lines.append(
            f"| {cat} | {c['count']} | {_fmt_pct(c['routing_acc'])} | {_fmt_pct(c['factual_acc'])} | {c['avg_latency_ms']:.1f} |"
        )
    lines.extend(
        [
            f"| **Overall** | **{stats['total']}** | **{_fmt_pct(stats['overall_routing_acc'])}** | **{_fmt_pct(stats['overall_factual_acc'])}** | **{stats['overall_avg_latency_ms']:.1f}** |",
            "",
        ]
    )
    print("\n".join(lines))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "category",
        "query",
        "node_path",
        "tools_called",
        "routing_correct",
        "factual_score",
        "is_hallucinated_auto",
        "factual_reason",
        "min_max_sim",
        "avg_max_sim",
        "unsupported_ratio",
        "retrieve_context_preview",
        "is_hallucinated_judge",
        "judge_reason",
        "latency_ms",
        "actual_output",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


@dataclass
class EvalOptions:
    csv_path: str = "evaluation_results.csv"
    skip_factual: bool = False
    sim_threshold: float = 0.62
    max_unsupported_ratio: float = 0.50


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LangGraph triage system (50 queries).")
    parser.add_argument("--csv", default="evaluation_results.csv", help="Output CSV path")
    parser.add_argument("--skip-judge", action="store_true", help="Skip factual scoring (routing + latency only)")
    parser.add_argument("--sim-threshold", type=float, default=0.62, help="Evidence cosine similarity threshold (vertical)")
    parser.add_argument(
        "--max-unsupported-ratio",
        type=float,
        default=0.50,
        help="Max fraction of answer sentences allowed below sim-threshold (vertical)",
    )
    args = parser.parse_args()
    opts = EvalOptions(
        csv_path=args.csv,
        skip_factual=args.skip_judge,
        sim_threshold=float(args.sim_threshold),
        max_unsupported_ratio=float(args.max_unsupported_ratio),
    )

    dataset = build_test_dataset()

    try:
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
    except Exception as e:
        logger.error("Failed to init LLM: %s", e)
        return 1

    ensure_chroma_seed(llm_embedding)

    tool_names = {t.name for t in get_tools(llm_embedding)}
    if "multiply" not in tool_names:
        logger.warning(
            "Tool 'multiply' is not registered. Utility routing checks expect multiply; "
            "add it in utils/tools_config.py for meaningful Utility scores."
        )

    tools = get_tools(llm_embedding)
    tool_config = ToolConfig(tools)

    db_connection_pool = None
    try:
        connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
        db_connection_pool = ConnectionPool(
            conninfo=Config.DB_URI,
            max_size=20,
            min_size=2,
            kwargs=connection_kwargs,
            timeout=10,
        )
        db_connection_pool.open()
        logger.info("Database connection pool initialized for evaluation")
        monitor_connection_pool(db_connection_pool, interval=120)
    except Exception as e:
        logger.warning("Database unavailable, using in-memory store/checkpointer: %s", e)
        db_connection_pool = None

    try:
        graph = create_graph(db_connection_pool, llm_chat, llm_embedding, tool_config)
    except ConnectionPoolError as e:
        logger.error("create_graph failed: %s", e)
        return 1

    result_rows: List[Dict[str, Any]] = []
    rout_ok = 0
    fact_ok = 0
    latencies: List[float] = []

    by_cat: Dict[str, Dict[str, Any]] = {
        "general": {"count": 0, "r_ok": 0, "f_ok": 0, "f_judged": 0, "lat": []},
        "utility": {"count": 0, "r_ok": 0, "f_ok": 0, "f_judged": 0, "lat": []},
        "vertical": {"count": 0, "r_ok": 0, "f_ok": 0, "f_judged": 0, "lat": []},
    }

    for item in dataset:
        cid = item["id"]
        cat = item["category"]
        q = item["query"]
        exp = item["expected_facts"]

        cfg = {
            "configurable": {
                "thread_id": f"eval-{uuid.uuid4().hex}",
                "user_id": f"eval-user-{cid}",
            },
            "recursion_limit": 50,
        }

        try:
            path, tools_called, output, latency_sec, retrieved_texts = run_graph_once(graph, q, cfg)
        except Exception as e:
            logger.exception("Graph run failed for %s: %s", cid, e)
            path, tools_called, output = [], [], ""
            latency_sec = 0.0
            retrieved_texts = []

        r_ok = routing_correct(cat, path, tools_called)
        if r_ok:
            rout_ok += 1
        by_cat[cat]["count"] += 1
        if r_ok:
            by_cat[cat]["r_ok"] += 1

        lat_ms = latency_sec * 1000.0
        latencies.append(lat_ms)
        by_cat[cat]["lat"].append(lat_ms)

        factual_score = 0
        factual_reason = ""
        is_hall_str = ""
        min_max_sim = float("nan")
        avg_max_sim = float("nan")
        unsupported_ratio = float("nan")
        retrieve_preview = ""

        if retrieved_texts:
            retrieve_preview = (retrieved_texts[-1] or "").replace("\r\n", "\n")[:500]

        if opts.skip_factual:
            factual_score = -1
            factual_reason = "SKIPPED"
        else:
            diag: Dict[str, Any] = {}
            if cat == "utility":
                factual_score, factual_reason, diag = score_utility(exp, output or "")
            elif cat == "general":
                factual_score, factual_reason, diag = score_general(output or "")
            else:
                factual_score, factual_reason, diag = score_vertical(
                    retrieved_texts=retrieved_texts,
                    actual_output=output or "",
                    llm_embedding=llm_embedding,
                    llm_chat=llm_chat,
                    sim_threshold=opts.sim_threshold,
                    max_unsupported_ratio=opts.max_unsupported_ratio,
                )
                if "min_max_sim" in diag:
                    min_max_sim = float(diag.get("min_max_sim"))
                if "avg_max_sim" in diag:
                    avg_max_sim = float(diag.get("avg_max_sim"))
                if "unsupported_ratio" in diag:
                    unsupported_ratio = float(diag.get("unsupported_ratio"))

            is_hall_str = "true" if factual_score == 0 else ("false" if factual_score == 1 else "")
            if factual_score in (0, 1):
                by_cat[cat]["f_judged"] += 1
                if factual_score == 1:
                    fact_ok += 1
                    by_cat[cat]["f_ok"] += 1

        result_rows.append(
            {
                "id": cid,
                "category": cat,
                "query": q,
                "node_path": " -> ".join(path),
                "tools_called": ",".join(tools_called),
                "routing_correct": int(r_ok),
                "factual_score": factual_score,
                "is_hallucinated_auto": is_hall_str,
                "factual_reason": factual_reason[:500],
                "min_max_sim": "" if math.isnan(min_max_sim) else f"{min_max_sim:.4f}",
                "avg_max_sim": "" if math.isnan(avg_max_sim) else f"{avg_max_sim:.4f}",
                "unsupported_ratio": "" if math.isnan(unsupported_ratio) else f"{unsupported_ratio:.4f}",
                "retrieve_context_preview": retrieve_preview,
                "is_hallucinated_judge": is_hall_str,
                "judge_reason": factual_reason[:500],
                "latency_ms": f"{lat_ms:.2f}",
                "actual_output": (output or "").replace("\r\n", "\n"),
            }
        )

    n = len(dataset)
    overall_r = rout_ok / n if n else 0.0
    overall_f_judged = sum(int(by_cat[c]["f_judged"]) for c in ("general", "utility", "vertical"))
    overall_f = (fact_ok / overall_f_judged) if (overall_f_judged and not opts.skip_factual) else 0.0
    overall_lat = sum(latencies) / len(latencies) if latencies else 0.0

    stats_out: Dict[str, Any] = {
        "total": n,
        "overall_routing_acc": overall_r,
        "overall_factual_acc": overall_f if (not opts.skip_factual and overall_f_judged) else float("nan"),
        "overall_avg_latency_ms": overall_lat,
        "by_category": {},
    }

    for cat in ("general", "utility", "vertical"):
        bc = by_cat[cat]
        c = bc["count"] or 1
        fj = bc["f_judged"] or 0
        avg_l = sum(bc["lat"]) / len(bc["lat"]) if bc["lat"] else 0.0
        stats_out["by_category"][cat] = {
            "count": bc["count"],
            "routing_acc": bc["r_ok"] / c,
            "factual_acc": (bc["f_ok"] / fj) if (not opts.skip_factual and fj) else float("nan"),
            "avg_latency_ms": avg_l,
        }

    print_markdown_report(stats_out)
    if opts.skip_factual:
        print("Note: Factual scoring was skipped (--skip-judge). Overall factual column is N/A.\n")

    write_csv(opts.csv_path, result_rows)
    print(f"Detailed rows written to: {opts.csv_path}\n")

    if db_connection_pool and not db_connection_pool.closed:
        try:
            db_connection_pool.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
