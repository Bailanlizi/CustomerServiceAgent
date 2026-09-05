# 🤖 E-commerce Smart Agent v4.0：可审计的电商客服 Agent

## 🌟 项目定位

电商客服 AI 最普遍的通病是**虚假承诺 / 过度承诺**：模型给出"AI 说包退、人工说不行"的答复，无法给出政策依据，出了问题也无法追溯责任。本项目是一个针对该痛点的**可审计客服 Agent**：

- **依据真实**：回答只能基于向量检索召回的政策条款，检索层全链路保留条款编号（`clause_ids`）与来源信息；
- **引用可校验**：政策回答采用结构化生成（`answer + applied_clause_ids + evidence_clause_ids`），后端执行确定性校验 `applied ⊆ evidence ⊆ 检索条款集合`，非法引用在输出前被拦截；
- **失败安全降级**：校验不通过时携带错误原因重试一次，仍失败则返回固定安全答复，绝不让未经校验的承诺触达用户。

在此基础上，系统同时提供完整的客服业务闭环：订单查询、政策咨询、退货退款申请（工具调用 + 统一人工审核）、人工审核工作台，并用一套 45 题的条款级评测集对 RAG 质量做量化度量。

## 🚀 主要特性

*   **政策回答 Guardrail（核心亮点）**：结构化生成 + 确定性引用校验 + 安全降级，从机制上堵死"编造条款号 / 检索不到仍作承诺"两类风险（详见下节）。
*   **来源感知检索排序**：政策文档按 4 层来源建模（正式条款 > FAQ，FAQ 通过 canonical 映射关联正式条款），检索结果带相似度阈值过滤（distance < 0.5）与权威重排。
*   **智能问答**：基于 LLM 的订单查询与政策咨询，意图识别采用结构化输出四分类（ORDER / POLICY / REFUND / OTHER）。
*   **退货申请流程**：退款 Agent 自主选择受控工具（资格预检 / 提交申请 / 进度查询），缺参数时主动向用户索要；用户身份由 `InjectedState` 注入，越权查询被数据库层拦截。
*   **退款安全闭环（资金操作全人工 + 幂等防重）**：所有退款申请不分金额统一进入人工审核，资金移动必须管理员批准；订单级唯一约束杜绝重复退款，支付任务以条件更新实现幂等抢占与卡死自动恢复，全程写入审计日志。
*   **工具能力治理（ToolRegistry / GuardedToolExecutor / ToolOutcome）**：所有工具（含 LangChain `@tool` 薄壳）经统一的 `ToolCapabilityRegistry` 注册与 `GuardedToolExecutor` 路由，Guard 按域 / 阶段 / 槽位 / 资格 / 用户确认 5 步确定性校验，模型误调用直接结构化拒绝；`ToolOutcome` 由 Registry 自动 envelope 9 项审计元数据（`tool_name` / `domain` / `workflow_stage` / `conversation_id` / `thread_id` / `user_id` / `order_id` / `timestamp` / `idempotency_key`），FSM / 审计 / 管理员队列共享同一字段源；写工具按 `idempotency_key_template` 在 Registry 进程内缓存成功 outcome（双层短路之一），管理员审计列表用 `WHERE thread_id IN (...)` 批量加载 `ConversationSession` 消除 N+1。
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

## 🛡️ 退款安全（资金操作全人工 + 幂等防重）

退款涉及资金移动，系统以「Agent 只采集信息与创建申请，资金移动必须人工批准 + 幂等执行」为原则，构建四层防护：

- **申请层（防重复）**：`refund_applications.order_id` 物理唯一约束 + 服务层全状态拦截，同一订单只允许一个退款案件，并发提交由 `IntegrityError` 兜底。
- **审批层（防越权 + 防并发）**：管理端接口改用数据库回查 `users.is_admin/is_active` 鉴权（JWT 不再签发角色 claim），审批在事务内以行锁 + 条件更新原子执行，两个管理员同时批准只有一次成功；拒绝必须携带非空审核理由。
- **支付层（防重复执行）**：支付任务只接收 `refund_id`、金额从数据库读取，通过条件更新 `APPROVED → PROCESSING → COMPLETED` 实现原子抢占；`PROCESSING` 卡死由 Celery Beat 每分钟扫描恢复为 `APPROVED` 并重新投递。
- **审计层（可追溯）**：所有退款申请（不分金额）统一创建 `AuditLog`，审核快照含订单号、状态、金额与商品明细，状态流转全程留痕。

**当前边界（如实声明）**：退款支付为 mock 实现（模拟打印 + 延时），尚未接入真实支付网关。接入真实支付前必须补 `RefundPayment` 支付流水表、网关幂等键与第三方交易号持久化，否则 Celery 重试可能造成真实重复打款。

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
│   │   ├── tool_registry.py # ToolCapability Registry + GuardedToolExecutor + ToolOutcome + 审计钩子
│   │   ├── tools.py # core_* handler + LangChain @tool 薄壳（薄壳路由到 Registry）
│   │   ├── workflow.py # 工作流编排与编译
│   │   └── workflows/ # 领域子图
│   │       ├── refund.py # RefundWorkflow（IDLE→IDENTIFY_ORDER→COLLECT_REASON→ELIGIBILITY_CHECKED→WAITING_CONFIRMATION→SUBMITTED）
│   │       └── order.py # OrderWorkflow（query_order → 自合成 answer，P3 新建）
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
