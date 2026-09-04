# 🤖 E-commerce Smart Agent v4.0：可审计的电商客服 Agent

## 🌟 项目定位

电商客服 AI 最普遍的通病是**虚假承诺 / 过度承诺**：模型给出"AI 说包退、人工说不行"的答复，无法给出政策依据，出了问题也无法追溯责任。本项目是一个针对该痛点的**可审计客服 Agent**：

- **依据真实**：回答只能基于向量检索召回的政策条款，检索层全链路保留条款编号（`clause_ids`）与来源信息；
- **引用可校验**：政策回答采用结构化生成（`answer + applied_clause_ids + evidence_clause_ids`），后端执行确定性校验 `applied ⊆ evidence ⊆ 检索条款集合`，非法引用在输出前被拦截；
- **失败安全降级**：校验不通过时携带错误原因重试一次，仍失败则返回固定安全答复，绝不让未经校验的承诺触达用户。

在此基础上，系统同时提供完整的客服业务闭环：订单查询、政策咨询、退货退款申请（工具调用 + 风险分级转人工）、人工审核工作台，并用一套 45 题的条款级评测集对 RAG 质量做量化度量。

## 🚀 主要特性

*   **政策回答 Guardrail（核心亮点）**：结构化生成 + 确定性引用校验 + 安全降级，从机制上堵死"编造条款号 / 检索不到仍作承诺"两类风险（详见下节）。
*   **来源感知检索排序**：政策文档按 4 层来源建模（正式条款 > FAQ，FAQ 通过 canonical 映射关联正式条款），检索结果带相似度阈值过滤（distance < 0.5）与权威重排。
*   **智能问答**：基于 LLM 的订单查询与政策咨询，意图识别采用结构化输出四分类（ORDER / POLICY / REFUND / OTHER）。
*   **退货申请流程**：退款 Agent 自主选择受控工具（资格预检 / 提交申请 / 进度查询），缺参数时主动向用户索要；用户身份由 `InjectedState` 注入，越权查询被数据库层拦截。
*   **智能风控与人工审核**：大额退款等高风险申请自动转交管理员人工审核。
*   **实时状态同步**：通过 WebSocket 实现用户和管理员界面的实时状态更新。
*   **管理员工作台**：Gradio 构建的 B 端界面，支持任务队列、会话回放、一键决策。
*   **异步任务处理**：Celery 处理退款支付、短信通知等耗时操作。
*   **RAG 评测体系**：45 题条款级测试集，覆盖 6 个难度维度，度量 hit@5 / clause_recall@5 / MRR@5 / 引用精确率及 RAGAS faithfulness（judge 模型与业务模型隔离）。

## 🛡️ 政策回答 Guardrail 闭环

POLICY 意图路径的完整证据链（`app/graph/nodes.py` + `app/services/policy_answer_guard.py`）：

```text
用户问题
   │
   ▼
retrieve 节点：向量检索 top-20 → 阈值过滤 → 权威重排 → top-5
   │  结构化证据（正文 / clause_ids / canonical_clause_ids / 来源 / 排名 / 距离）完整写入 state
   ▼
无证据？ ── 是 ──► 直接返回"暂未查询到相关规定"（不调用 LLM，零成本短路）
   │ 否
   ▼
生成节点：with_structured_output 产出 {answer, applied_clause_ids, evidence_clause_ids}
   │
   ▼
确定性校验（纯代码，不依赖模型自觉）：
   1. applied ⊆ evidence（实际适用条款必须是证据子集）
   2. evidence ⊆ 检索返回的 clause_ids ∪ canonical_clause_ids
   3. 用户可见 answer 中不得出现任何条款编号
   4. 有证据时引用列表不得为空
   │
   ▼
校验失败 ──► 携带具体错误原因重试一次 ── 仍失败 ──► 固定安全答复（fallback）
   │ 通过
   ▼
答案经 SSE 整体下发（结构化生成调用打 policy_guard 标签，中间 token 一律不外泄）
```

每轮回答的审计记录（状态、applied / evidence / allowed 三元组、重试次数）随 LangGraph Redis checkpoint 持久化，可事后查证每条承诺的依据。

**当前边界（如实声明）**：Guardrail 保证的是"引用合法、可追溯"，不校验自然语言语义是否被条款完全支持；审计记录暂存于会话 checkpoint（随会话过期），尚未落独立审计表。

### 多轮会话历史的一致性

`AgentState.messages` 经 Redis checkpoint 跨轮累积，消费它的是退款 Tool Agent，因此历史必须是**合法的 Human/AI 交替序列**——否则模型会把历史中"没有回复过的用户消息"当作待答问题，一次性复述作答。为此锁定两条不变量（回归见 `test/test_conversation_history.py`）：

1. **每轮必回写**：订单查询 / 政策咨询走 `generate` 节点，除返回 `answer` 外必须以 `AIMessage` 回写本轮回复，保证每条用户消息都有配对的助手回复。
2. **历史窗口受控**：退款 Agent 最多携带最近 3 轮（`MAX_REFUND_HISTORY_TURNS`），且窗口起点固定为一条 `HumanMessage`——否则会把带 `tool_calls` 的 `AIMessage` 与其配对的 `ToolMessage` 拆开，触发模型 API 报错。

窗口只裁剪跨意图的无关历史（如之前的订单查询、政策咨询），退款流程自身的连续性（"我要退 SN20240001" → "质量问题，鞋底开胶"）不受影响。

## 🏗️ 项目结构

```text
├── app # 主应用目录
│   ├── api # API 接口定义
│   │   └── v1
│   │       ├── admin.py # 管理员 API (获取任务, 决策)
│   │       ├── chat.py # 聊天接口 (SSE 流式返回, Guardrail 标签过滤)
│   │       ├── schemas.py # 请求/响应 Pydantic 模型
│   │       ├── status.py # 状态查询 API
│   │       └── websocket.py # WebSocket 连接端点
│   ├── core # 配置 / 数据库 / JWT 认证
│   ├── evaluation # 评测模块
│   │   ├── metrics.py # 条款级检索与引用指标 (hit/recall/MRR/precision)
│   │   └── chinese_ragas_prompts.py # RAGAS faithfulness 中文化 judge prompt
│   ├── frontend # Gradio 前端 (用户聊天 + 管理员工作台)
│   ├── graph # LangGraph 核心逻辑
│   │   ├── nodes.py # 节点定义 (意图路由, 检索, 生成, 退款 Agent, 结构化政策生成)
│   │   ├── state.py # 图状态 (含 policy_evidence / policy_answer_audit)
│   │   ├── tools.py # 退款工具 (资格预检, 提交申请, 进度查询)
│   │   └── workflow.py # 工作流编排与编译
│   ├── models # SQLModel ORM (订单, 知识库块, 退款, 审计, 消息卡片)
│   ├── services # 业务服务层
│   │   ├── policy_answer_guard.py # 政策回答引用校验 (确定性 Guardrail)
│   │   ├── policy_chunks.py # 政策文档条款级解析与标注
│   │   ├── policy_retrieval.py # 向量检索 + 来源感知权威重排
│   │   └── refund_service.py # 退款业务逻辑
│   ├── tasks # Celery 异步任务 (退款支付, 短信, 管理员通知)
│   └── main.py # FastAPI 入口
├── data # 政策知识库源文档 (6 个 Markdown, 含条款编号与优先级标注)
├── eval # 评测资产
│   ├── testset.json # 45 题条款级测试集 (6 难度维度 × 标注期望条款)
│   └── runs # 评测运行产物 (JSON)
├── scripts
│   ├── etl_policy.py # 知识库 ETL (解析 + Embedding 入库)
│   ├── run_rag_baseline.py # 检索基线评测 (hit/recall/MRR/precision)
│   ├── evaluate_ragas.py # RAGAS faithfulness 评测 (judge 隔离)
│   └── seed_data.py / seed_large_data.py # 种子数据
├── test # 单元与回归测试 (Guardrail 校验, SSE, 条款解析, 退款规则/工具等)
├── docker-compose.yaml # PostgreSQL(pgvector) + Redis + FastAPI + Celery
├── migrations # Alembic 迁移脚本
├── start.sh / start_worker.sh # 启动脚本
```

## 🛠️ 技术栈

*   **Python / FastAPI**：REST API + WebSocket 服务。
*   **LangChain / LangGraph**：Agent 编排、意图识别、RAG、多步骤工作流；Redis 作为会话 Checkpointer。
*   **SQLModel + PostgreSQL (pgvector)**：数据模型与向量存储。
*   **结构化输出（`with_structured_output`）**：意图分类与政策回答均使用受限 schema，不依赖自由文本解析。
*   **Redis**：缓存、Celery broker、LangGraph checkpoint。
*   **Celery**：退款支付、短信通知等异步任务。
*   **Gradio**：用户聊天界面与管理员工作台。
*   **JWT (PyJWT)**：认证与授权。
*   **OpenAI API / Qwen (通义千问)**：LLM 与 Embedding（适配器接入）。
*   **RAGAS**：faithfulness 评测（judge 模型与业务模型隔离，避免自评偏置）。
*   **Docker / Docker Compose / Alembic**：容器化部署与数据库迁移。

## 📊 评测体系

测试集 `eval/testset.json` 共 45 题，按 6 个难度维度设计（层级冲突、多跳综合、品类边界、抗幻觉、量化细节、FAQ），每题标注期望命中条款，`field_spec` 约定首个来源为主权威条款。

**当前基线**（评测产物见 `eval/runs/`）：

| 指标 | 数值 | 说明 |
|---|---|---|
| primary_hit@5 / any_hit@5 | 1.000 / 1.000 | 期望条款全部进入 top-5 |
| clause_recall@5 | 0.9815 | 标注条款覆盖率 |
| MRR@5 | 0.8685 | 排序质量，短板集中在层级冲突类问题 |
| answer_primary_hit | 0.9556 | 回答中正确引用主权威条款 |
| faithfulness (RAGAS) | 0.9091 | 回答对检索证据的忠实度 |

复现方式：

```bash
python scripts/run_rag_baseline.py   # 检索侧: hit / recall / MRR / precision
python scripts/evaluate_ragas.py     # 生成侧: RAGAS faithfulness
```

## ⚡ 快速开始

本项目使用 **uv** 管理依赖（非 Poetry），`.venv` 由 `uv sync` 创建。

```bash
# 1. 启动基础设施 (PostgreSQL+pgvector, Redis)
docker-compose up -d db redis

# 2. 安装依赖 (生成/复用 .venv)
uv sync

# 3. 配置 .env (LLM/Embedding 的 API Key 等), 然后数据库迁移 + 知识库入库 + 种子数据
uv run alembic upgrade head
uv run python scripts/etl_policy.py
uv run python scripts/seed_data.py

# 4. 启动服务
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
uv run celery -A app.celery_app worker --loglevel=info --pool=solo
```

访问地址：API `http://localhost:8000`（文档 `/docs`）· 用户界面 `http://localhost:7860` · 管理员工作台 `http://localhost:7861`

## 📸 界面演示

### 订单查询
<img src="assets/image/order_query.png" width="600" alt="订单查询" />

### 退货申请
<img src="assets/image/refund_apply.png" width="600" alt="退货申请" />

### 政策咨询
<img src="assets/image/policy_ask.png" width="600" alt="政策咨询" />

### 意图识别
<img src="assets/image/intent_detect.png" width="600" alt="意图识别" />

### 非法查询他人订单（越权拦截）
<img src="assets/image/illegal_query.png" width="600" alt="非法查询" />
