# 基于 LangGraph 的智能分诊系统（Agentic RAG）

本项目实现了一个基于 LangGraph 的智能分诊/问答系统：
- **闲聊/通用对话**：直接由模型回答，不触发工具。
- **计算类请求**：自动调用 `multiply` 工具并返回结果。
- **垂直领域检索问答**：自动调用 `retrieve` 从本地 Chroma 向量库检索证据，通过 `grade_documents` 评估相关性，并在 `generate` 阶段生成最终回答。

项目同时提供：
- FastAPI 后端接口（OpenAI ChatCompletions 风格）。
- Gradio WebUI 前端。
- 两套离线评测：固定测试集评测 + 合成数据集（LLM-as-judge）评测。

## 目录结构

```
L1-Project-2/
  main.py                         # FastAPI 后端
  webUI.py                        # Gradio WebUI
  ragAgent.py                     # LangGraph 工作流与节点逻辑
  utils/
    config.py                     # 统一配置（向量库路径、DB_URI、LLM_TYPE 等）
    llms.py                       # OpenAI 兼容接口的 LLM/Embedding 初始化
    tools_config.py               # 工具注册：retrieve + multiply
    pdfSplitTest_Ch.py            # PDF 中文切分
    pdfSplitTest_En.py            # PDF 英文切分
  prompts/                        # agent/grade/rewrite/generate 提示词
  input/
    健康档案.pdf                   # 示例健康档案
    deepseek-v3-1-4.pdf           # 示例论文
  chromaDB/                       # Chroma 持久化目录（已包含示例数据时可直接用）
  docker-compose.yml              # 可选：PostgreSQL 连接池运行环境
  evaluate_system.py              # 固定测试集评测（50 条）
  evaluate_synthetic_LLM.py       # 合成评测：生成问题 + 运行 + LLM 评审
  *.csv                           # 评测输出
  graph.png                       # 图结构可视化（如已生成）
```

## 核心能力与工作流

系统以 LangGraph 状态图为核心，关键节点（名称会出现在评测的 `node_path` 中）：
- `agent`：识别意图并决定是否调用工具。
- `call_tools`：并行执行工具（检索/计算）。
- `grade_documents`：对检索结果做相关性评估，并决定是否进入重写循环。
- `rewrite`：对问题进行自我改写（受 `rewrite_count` 递归限制控制）。
- `generate`：基于证据生成最终回复。

工具由 [tools_config.py](file:///d:/7045NLP/%E9%A1%B9%E7%9B%AE2_%E5%9F%BA%E4%BA%8ELangGraph%E5%AE%9E%E7%8E%B0%E6%99%BA%E8%83%BD%E5%88%86%E8%AF%8A%E7%B3%BB%E7%BB%9F/L1-Project-2/utils/tools_config.py) 注册：
- `retrieve`：Chroma 检索（默认 `k=8`）。
- `multiply`：乘法计算。

## 环境变量与配置

项目配置集中在 [config.py](file:///d:/7045NLP/%E9%A1%B9%E7%9B%AE2_%E5%9F%BA%E4%BA%8ELangGraph%E5%AE%9E%E7%8E%B0%E6%99%BA%E8%83%BD%E5%88%86%E8%AF%8A%E7%B3%BB%E7%BB%9F/L1-Project-2/utils/config.py) 和 [llms.py](file:///d:/7045NLP/%E9%A1%B9%E7%9B%AE2_%E5%9F%BA%E4%BA%8ELangGraph%E5%AE%9E%E7%8E%B0%E6%99%BA%E8%83%BD%E5%88%86%E8%AF%8A%E7%B3%BB%E7%BB%9F/L1-Project-2/utils/llms.py)。常用环境变量：

- `LLM_TYPE`：`openai` / `qwen` / `oneapi` / `ollama`
- `OPENAI_API_KEY`：当 `LLM_TYPE=openai` 时需要
- `OPENAI_BASE_URL`：OpenAI/兼容服务地址（可选）
- `DASHSCOPE_API_KEY`：当 `LLM_TYPE=qwen` 时需要
- `DB_URI`：PostgreSQL 连接串（可选；不提供时会退化为内存存储）

提示：`docker-compose.yml` 中提供了一个 Postgres 示例，但其中用户/健康检查用户存在不一致，若使用请自行统一，并确保 `DB_URI` 与实际账号匹配。

## 安装依赖

仓库未提供 `requirements.txt` / `pyproject.toml`。你可以根据导入依赖安装（示例，按需取用）：

```bash
pip install langgraph langchain-core langchain-openai langchain-chroma chromadb \
  fastapi uvicorn gradio psycopg2-binary psycopg-pool tenacity concurrent-log-handler
```

如果你使用 `ollama` 或其他兼容服务，请确保对应服务已启动并可通过 `OPENAI_BASE_URL` 访问。

## 启动与使用

### 1) 启动后端（FastAPI）

在 `L1-Project-2` 目录下运行：

```bash
python main.py
```

默认服务监听由 `Config.HOST/Config.PORT` 控制（默认 `0.0.0.0:8012`）。

### 2) 启动前端（Gradio）

```bash
python webUI.py
```

WebUI 默认请求后端：`http://localhost:8012/v1/chat/completions`。

### 3) 向量库准备（可选）

系统运行与评测均依赖 Chroma 向量库。
- 如果 `chromaDB/` 已包含数据，可直接运行。
- 如果为空：
  - `evaluate_system.py` 在评测前会尝试将 `input/健康档案.pdf` 自动播种到 Chroma。
  - 也可使用 `vectorSave.py` 手动灌库（注意：请仅通过环境变量提供密钥，避免在代码中写入任何密钥）。

## 评测

### 1) 固定测试集评测（50 条）

脚本：`evaluate_system.py`。

特点：
- **Routing Accuracy**：通过 `node_path` + 工具调用检查是否走对分支。
- **Factual/Hallucination（自动）**：
  - Utility：抽取数值并比对乘积是否正确。
  - General：检测闲聊中是否编造“患者名 + 医疗事实/数值”。
  - Vertical：将回答拆句，与检索证据切块做 embedding 相似度对齐（可调阈值）。

运行示例：

```bash
python evaluate_system.py
python evaluate_system.py --csv evaluation_results.csv
python evaluate_system.py --skip-judge
python evaluate_system.py --sim-threshold 0.62 --max-unsupported-ratio 0.50
```

输出：生成 CSV（包含 `node_path/tools_called/routing_correct/factual_score/latency_ms/actual_output` 等字段）。

### 2) 合成评测（LLM-as-judge）

脚本：`evaluate_synthetic_LLM.py`。

方法概述：
- 从 Chroma 抽样原文 chunk。
- 由模型为每个 chunk 生成 3 个“仅凭该 chunk 可回答”的问题。
- 运行系统并记录检索/路径/输出。
- 使用高级模型作为 judge 对 `RETRIEVED DOCUMENTS` 与回答做 **faithfulness(0/1)** 评审。

运行示例：

```bash
python evaluate_synthetic_LLM.py --full --max-chunks 10 --csv evaluation_synthetic_results_LLM_judge.csv
python evaluate_synthetic_LLM.py --generate-only --max-chunks 10 --dataset synthetic_test_dataset.json
python evaluate_synthetic_LLM.py --eval-only --dataset synthetic_test_dataset.json --csv evaluation_synthetic_results_LLM_judge.csv
python evaluate_synthetic_LLM.py --eval-only --skip-faithfulness-judge
```

输出 CSV 字段包括：
`chunk_id, question, node_path, tools_called, overlap_source_vs_retrieved, routing_vertical_ok, faithfulness, faithfulness_reason, latency_ms, assistant_output`。

## 常见问题（Troubleshooting）

- **数据库不可用**：后端与评测会自动降级为内存存储（仍可运行，但不会持久化会话/检查点）。
- **检索为空/命中差**：检查 `chromaDB/` 是否存在并与 `Config.CHROMADB_COLLECTION_NAME` 一致；必要时重新播种。
- **密钥/地址配置**：确保 `LLM_TYPE` 对应的 `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` 与 `OPENAI_BASE_URL` 已正确设置。

## License

见 [LICENSE](file:///d:/7045NLP/%E9%A1%B9%E7%9B%AE2_%E5%9F%BA%E4%BA%8ELangGraph%E5%AE%9E%E7%8E%B0%E6%99%BA%E8%83%BD%E5%88%86%E8%AF%8A%E7%B3%BB%E7%BB%9F/L1-Project-2/LICENSE)。

