# ai-tech-agent

## 项目简介
`ai-tech-agent` 是一个面向教学场景的 AI 服务，基于 `FastAPI + RAG + 工具调用 + SSE 流式聊天` 构建。

## 项目矩阵（AI自适应学习引擎）

- 前端：[ai-tech-vue](https://github.com/organwalk/ai-tech-vue)
- 后端：[ai-tech](https://github.com/organwalk/ai-tech)
- Agent（当前仓库）：[ai-tech-agent](https://github.com/organwalk/ai-tech-agent)

它主要用于以下场景：
- 教学知识库入库与检索问答
- 学习指南生成
- 题目生成与试卷分析
- 学习诊断与学习行为分析
- 教师/学员双角色对话（含工具增强）
- RAG 检索效果评测

## 核心功能
- 知识入库：下载并解析 PDF / DOCX / 文本，分块后向量化写入 ChromaDB。
- 学习指南：结合给定主题与知识库片段生成结构化学习指南。
- 出题与分析：根据范围出题，支持试卷分析与结果解读。
- 学习诊断：基于历史内容生成 Markdown + JSON 双段诊断结果。
- 聊天问答：
  - 学员模式：知识检索 + 记忆检索 + SSE 流式回复
  - 教师模式：在学员模式基础上支持工具调用（外部教学系统查询）
- RAG 评测：对检索效果做样本校验、指标计算和门禁判定。

## 技术栈与依赖
- Python 3.10
- FastAPI / Starlette / Uvicorn
- Pydantic v2
- ChromaDB（本地持久化向量库）
- Volcengine Ark SDK（`volcengine-python-sdk`）
- requests
- pdfplumber
- python-docx
- python-dotenv

## 目录结构
```text
ai-tech-agent/
├─ main.py                 # FastAPI 入口
├─ config.py               # 统一配置与环境变量读取
├─ models.py               # Pydantic 请求模型
├─ routers/                # 路由层（HTTP API）
├─ services/               # 业务与集成服务（LLM/RAG/Tool/Eval）
├─ prompts/                # Prompt 模板与构造函数
├─ scripts/                # RAG 数据集构建与评测脚本
├─ tests/                  # 单元测试
├─ docs/                   # 文档与示例数据集
├─ utils/                  # 通用工具
├─ knowledge_db/           # ChromaDB 本地数据目录（运行产物）
└─ reports/                # 评测输出目录（运行产物）
```

## 快速启动
### 1. 创建并激活虚拟环境
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装依赖
当前仓库未提供 `requirements.txt`，可先安装最小运行依赖：

```powershell
pip install fastapi uvicorn pydantic python-dotenv chromadb requests pdfplumber python-docx volcengine-python-sdk
```

### 3. 配置环境变量
在项目根目录创建 `.env`（示例）：

```env
ARK_API_KEY=YOUR_REAL_KEY
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL_ID=deepseek-v3-2-251201
ARK_EMBED_MODEL_ID=ep-xxxxxxxxxxxxxxxx
```

### 4. 启动服务
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 健康检查
- `GET /`：服务启动信息
- `GET /health`：健康状态

## 环境变量说明
以下变量来自 `config.py`（默认值为代码中的回退值）：

| 分类 | 变量名 | 默认值 | 说明 |
|---|---|---|---|
| Ark | `ARK_API_KEY` | `""` | LLM/Embedding 鉴权密钥 |
| Ark | `ARK_BASE_URL` | `https://ark.cn-beijing.volces.com/api/v3` | Ark API 地址 |
| Ark | `ARK_MODEL_ID` | `deepseek-v3-2-251201` | 对话模型 |
| Ark | `ARK_EMBED_MODEL_ID` | `ep-20260225005055-qq86c` | 向量模型 |
| Ark | `ARK_CLIENT_TIMEOUT` | `1800` | Ark 客户端超时（秒） |
| 存储 | `CHROMA_DB_PATH` | `<BASE_DIR>/knowledge_db` | ChromaDB 持久化路径 |
| 日志 | `LOG_LEVEL` | `INFO` | 日志级别 |
| 日志 | `LOG_PREVIEW_CHARS` | `220` | 日志预览截断长度 |
| 日志 | `ENABLE_RAG_VERBOSE_LOG` | `true` | 是否输出 RAG 详细日志 |
| 日志 | `ENABLE_TOOL_VERBOSE_LOG` | `true` | 是否输出工具调用详细日志 |
| LLM | `TOOL_CALL_TEMPERATURE` | `0.2` | 工具决策温度 |
| LLM | `FINAL_ANSWER_TEMPERATURE` | `0.7` | 最终回答温度 |
| LLM | `GUIDE_TEMPERATURE` | `0.7` | 指南生成温度 |
| LLM | `QUIZ_TEMPERATURE` | `0.8` | 出题温度 |
| LLM | `QUIZ_ANALYSIS_TEMPERATURE` | `0.8` | 试卷分析温度 |
| LLM | `DASHBOARD_ANALYSIS_TEMPERATURE` | `1.3` | 学情分析温度 |
| LLM | `DIAGNOSIS_TEMPERATURE` | `1.1` | 诊断温度 |
| LLM | `STREAM_DELAY_SECONDS` | `0.01` | SSE 分片发送延迟 |
| Tool | `TOOL_MAX_ROUNDS` | `5` | 工具循环最大轮次 |
| Tool | `TOOL_DUPLICATE_CALL_LIMIT` | `2` | 相同工具+参数重复上限 |
| Tool | `TOOL_CALL_TIMEOUT_SECONDS` | `10` | 工具 HTTP 超时 |
| Tool | `TOOL_DEFAULT_PAGE_NUM` | `1` | 默认分页页码 |
| Tool | `TOOL_DEFAULT_PAGE_SIZE` | `10` | 默认分页大小 |
| RAG | `RAG_DOC_TOP_K` | `8` | 文档检索 Top-K |
| RAG | `RAG_DOC_CANDIDATE_K` | `24` | 文档候选集大小 |
| RAG | `RAG_MEMORY_TOP_K` | `8` | 记忆检索 Top-K |
| RAG | `RAG_MEMORY_CANDIDATE_K` | `24` | 记忆候选集大小 |
| RAG | `RAG_SEMANTIC_WEIGHT` | `0.75` | 语义分权重 |
| RAG | `RAG_LEXICAL_WEIGHT` | `0.25` | 词法分权重 |
| RAG | `RAG_SCORE_FLOOR` | `0.0` | 最低分过滤阈值 |
| 解析 | `TEXT_CHUNK_SIZE` | `500` | 分块大小 |
| 解析 | `TEXT_CHUNK_OVERLAP` | `50` | 分块重叠 |
| Eval | `RAG_EVAL_TOP_K` | `8` | 评测 Top-K |
| Eval | `RAG_EVAL_CANDIDATE_K` | `24` | 评测候选集 |
| Eval | `RAG_EVAL_GATE_ENABLED` | `true` | 是否启用门禁 |
| Eval | `RAG_EVAL_GATE_HIT_RATE_MIN` | `0.50` | HitRate 下限 |
| Eval | `RAG_EVAL_GATE_MRR_MIN` | `0.30` | MRR 下限 |
| Eval | `RAG_EVAL_GATE_NDCG_MIN` | `0.40` | nDCG 下限 |

说明：工具注册表来自仓库内 JSON 文件（`tool_api_registry.json`、`student_tool_api_registry.json`），其中 `TOOL_JAVA_BASE_URL` 默认为 `http://127.0.0.1:8081`。

## API 概览
### 基础接口
- `GET /`：服务运行提示
- `GET /health`：健康检查

### 知识库
- `POST /api/v1/agent/knowledge/parse`
  - 请求字段：`file_id`, `file_url`, `window_id`
  - 功能：下载文件、解析文本、切块、向量化并入库

### 学习指南
- `POST /api/v1/agent/guide/generate`
  - 请求字段：`topic`, `file_ids[]`
  - 功能：基于主题与检索片段生成学习指南

### 学情/诊断
- `POST /api/v1/agent/dashboard/learner/analyze`
  - 请求字段：`system_prompt`, `context_data`
- `POST /api/v1/agent/diagnosis/self_study`
  - 请求字段：`system_prompt`, `user_content`
  - 输出含 `---MARKDOWN---` 与 `---JSON---`

### 出题与分析
- `POST /api/v1/agent/quiz/generate`
  - 请求字段：`system_prompt`, `user_data`, `file_ids[]`
- `POST /api/v1/agent/quiz/analysis`
  - 请求字段：`prompt`, `requirement`

### 对话
- `POST /api/v1/agent/chat/memory/add`
  - 请求字段：`window_id`, `chapter_id`, `msg_id`, `role`, `content`
- `POST /api/v1/agent/chat`
  - 请求字段：`query`, `windowId`, `chapterId`, `fileIds[]`, `history[]`
  - 响应：`text/event-stream`（SSE）
- `POST /api/v1/agent/chat/share`
  - 请求字段：`query`, `windowId`, `fileIds[]`, `history[]`, `userId`, `chapterId`, `token`
  - 响应：`text/event-stream`（SSE，含工具调用链路）

### RAG 评测
- `POST /api/v1/agent/rag/evaluate`
  - 请求字段：`samples[]`, `top_k`, `candidate_k`
  - 返回：`summary`, `validation_summary`, `gate`, `samples`

## RAG 评测工作流
按以下 4 步执行标准链路：

```powershell
# 1) 构建 seed 数据集
python scripts/build_rag_eval_seed.py --max-files 200 --samples-per-file 10 --seed 42 --output docs/rag_eval_seed_all_10.json

# 2) 构建 test_book 8x10 混合难度数据集
python scripts/build_rag_eval_test_book.py --seed-input docs/rag_eval_seed_all_10.json --output docs/rag_eval_test_book_8x10_mixed.json --review-output reports/rag_eval_test_book_8x10_spotcheck.json

# 3) 运行评测
python scripts/run_rag_eval.py --input docs/rag_eval_test_book_8x10_mixed.json --output reports/rag_eval_test_book_8x10_report.json

# 4) 生成 open/filter 拆分报告
python scripts/split_rag_eval_report.py --input reports/rag_eval_test_book_8x10_report.json --output reports/rag_eval_test_book_8x10_split.json
```

## 测试与验证
### 单元测试
```powershell
python -m unittest tests/test_rag_eval_service.py -v
```

### 常见失败原因
- `ModuleNotFoundError: No module named 'pydantic'`
  - 原因：未在项目虚拟环境安装依赖。
  - 处理：激活 `.venv` 后安装依赖再执行。
- `ConnectionError / timeout`（工具调用）
  - 原因：`TOOL_JAVA_BASE_URL` 对应服务不可达。
  - 处理：检查 Java 工具服务进程和网络连通性。
- `RAG retrieval empty`
  - 原因：`knowledge_db` 未入库或 `file_ids` 过滤不匹配。
  - 处理：先执行知识入库并校验 `file_id`。

## 安全规范（严格）
- 严禁提交真实密钥到仓库：`.env` 必须在 `.gitignore` 中。
- 密钥只允许本地注入或通过 CI Secret 注入。
- 一旦密钥泄露，必须立即在密钥管理平台轮换并废弃旧密钥。
- 提交前必须检查 `git diff` 与 `git status`，确认无敏感信息。
- 本项目当前规范要求：仓库中不得保留可直接使用的真实 `ARK_API_KEY`。

## 常见问题
### 1) 文本解析出现乱码
- 优先确认源文件编码与内容本身；非 UTF-8 文本建议先转码后入库。

### 2) 测试命令在系统 Python 下失败
- 请确认已激活项目虚拟环境：`.\.venv\Scripts\Activate.ps1`。

### 3) 启动成功但问答无检索上下文
- 检查是否先调用了 `/api/v1/agent/knowledge/parse` 完成入库。
- 检查请求中的 `fileIds` 与已入库 `file_id` 是否一致。

### 4) 教师端工具调用没有返回有效数据
- 检查 `tool_api_registry.json` 与 `student_tool_api_registry.json` 的 API 配置。
- 检查 `TOOL_JAVA_BASE_URL` 与 token 透传是否正确。

### 5) 报告目录过大
- `reports/` 与 `knowledge_db/` 属于运行产物，默认不纳入版本控制。

## 开发约束
- 不在文档中引入未落地的接口或参数。
- 业务接口、请求模型、响应结构以代码为准，文档仅做说明。
