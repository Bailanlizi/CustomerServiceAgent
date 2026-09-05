# 客服连续对话、记忆与受控编排架构

## 1. 目标与边界

本阶段解决客服 Agent 的“流程反复 / 话术循环”问题：用户在连续多轮对话中不应反复提供已确认的订单号、退款原因或诉求；系统应明确知道当前流程位置、已完成操作和下一步动作。

本设计仅覆盖连续对话、记忆、状态管理、工具治理和领域工作流拆分。退款仍沿用现有人工审批队列；**不**在本阶段实现人工完整接入实时对话、通用工单/SLA 系统、自由协商式多 Agent、LangGraph `interrupt()` 或通用 Plan-and-Execute。

## 2. 架构原则

1. 原始消息用于语言理解，结构化工作记忆用于流程决策，订单/退款/审核数据库记录用于业务事实；三者不能互相替代。
2. 对话流程状态、退款业务状态、审核动作分别维护，禁止形成双状态机。
3. LLM 负责理解、槽位抽取和受限话术；确定性代码负责状态迁移、工具前置条件、权限与副作用控制。
4. 只为确有多步骤和独立状态的领域建立子工作流；不为单步政策问答拆分工作流。
5. Orchestrator 必须保持薄：只读写工作记忆、选择领域工作流、处理安全的领域切换，不承载业务细节。

## 3. 目标架构

```text
Thin Orchestrator
  ├─ ConversationStateManager
  │   ├─ stable conversation_id
  │   ├─ working memory merge
  │   ├─ summary compaction
  │   └─ slot / pending action maintenance
  ├─ Domain Transition Resolver
  ├─ IntentRouter (仅无活跃领域或领域切换歧义时调用)
  ├─ OrderWorkflow (子图)
  ├─ PolicyWorkflow (保持现有 RAG + Guardrail 单节点)
  ├─ RefundWorkflow (子图)
  └─ ToolGuard + ToolCapability Registry
```

### 3.1 Thin Orchestrator

Orchestrator 的职责仅包括：

- 加载并合并当前 `ConversationSession` 与 LangGraph checkpoint；
- 根据 `active_domain`、当前输入和未决动作选择领域工作流；
- 在用户明确提出新目标时安全切换领域；
- 接收 workflow 的结构化 outcome 并写回工作记忆。

它不直接查询订单、不执行退款规则、不生成第二遍话术、不维护退款业务状态。目标实现规模为 50–100 行的调度逻辑，业务能力全部位于独立服务或子图中。

### 3.2 领域切换规则

`active_domain` 不能被当作永久锁。每轮输入按以下顺序处理：

```text
当前输入与活跃流程兼容？
  是：继续现有 workflow，不重新进行顶层意图分类。
  否：判断是否为明确的新目标。
       明确：保存旧流程摘要并切换 domain。
       不明确：提出最小澄清问题，不丢弃旧流程。
无 active_domain：调用现有 IntentRouter。
```

这避免“用户补充订单号时重新路由丢上下文”，也避免“用户从退款改问运费时被强行困在退款流程”。

## 4. 连续性记忆设计

### 4.1 ConversationSession

本阶段新增轻量会话实体，而不立即引入通用 Case/Ticket：

```text
conversation_id          稳定会话标识
user_id                  所属用户
client_session_id        客户端会话/渠道关联标识
active_domain            ORDER / POLICY / REFUND / null
working_memory_json      当前结构化工作记忆
conversation_summary     压缩后的已确认事实与未决事项
last_active_at           最近活动时间
```

`conversation_id` 必须由服务端生成并返回客户端。`thread_id` 继续用于 LangGraph checkpoint，但须稳定映射到 `conversation_id`；前端刷新或重新建立连接不能无条件生成全新业务会话。

### 4.2 四层记忆

| 层级 | 内容 | 真源 | 用途 |
| --- | --- | --- | --- |
| 短期消息 | 最近 Human / AI / Tool 消息 | LangGraph checkpoint | 指代理解、自然语言连贯性 |
| 工作记忆 | 当前目标、槽位、阶段、下一步 | ConversationSession | 流程推进与防重复提问 |
| 会话摘要 | 已确认事实、已做动作、未决事项 | ConversationSession | 长对话压缩、恢复与跨 domain 切换 |
| 业务事实 | 订单、退款、审核、政策证据 | PostgreSQL / 工具 / RAG | 权限校验、真实状态、可审计结论 |

原始聊天记录绝不作为订单状态、退款资格或管理员决定的业务真源。

### 4.3 Working Memory Schema

`AgentState` 与 `ConversationSession.working_memory_json` 采用同一逻辑结构：

```python
active_domain: Literal["ORDER", "POLICY", "REFUND"] | None
active_order_id: int | None
active_order_sn: str | None
conversation_goal: str | None
collected_slots: dict[str, object]
pending_slots: list[str]
workflow_stage: str | None
last_tool_result: dict | None
next_action: Literal["ASK", "TOOL_CALL", "COMPOSE", "WAIT"] | None
conversation_summary: str | None
```

更新规则：

- 仅可信来源可写入已确认槽位：数据库查询、通过校验的工具结果、用户明确回答。
- 新输入不能覆盖已确认的订单归属；订单切换必须显式识别并重新校验权限。
- 每个 workflow 返回结构化 outcome，由 `ConversationStateManager` 统一合并。
- 当消息轮数或 token 预算超过阈值时，压缩已完成阶段；不得压缩未决槽位、业务 ID、工具失败原因和用户明确约束。

## 5. 状态真源与退款流程

### 5.1 三类状态不可混用

| 状态 | 真源 | 说明 |
| --- | --- | --- |
| 对话流程阶段 | ConversationSession / AgentState | AI 正在收集什么、下一步该问什么或调用什么工具 |
| 退款业务状态 | RefundApplication | `PENDING → APPROVED → PROCESSING → COMPLETED`，以及 `REJECTED/CANCELLED` |
| 人工审核动作 | AuditLog.action | 审批队列中的 `PENDING / APPROVE / REJECT` |

用户可见的“等待审核”“退款处理中”“已完成”是根据业务状态和审核动作派生的展示状态，不能成为额外可写状态。

### 5.2 RefundWorkflow

退款是第一个需要显式 FSM 的领域子图：

```text
IDLE
  → IDENTIFY_ORDER
  → COLLECT_REASON
  → ELIGIBILITY_CHECKED
  → WAITING_CONFIRMATION
  → SUBMITTED
```

- `IDENTIFY_ORDER`：复用 `active_order_id`；缺失时只询问订单号。
- `COLLECT_REASON`：订单已确认但退款原因缺失时才询问原因。
- `ELIGIBILITY_CHECKED`：调用硬规则工具并将结果写入 `last_tool_result` 与 `collected_slots`。
- `WAITING_CONFIRMATION`：向用户展示订单、金额与原因，等待显式确认。第一版用前端/API 确认字段或按钮，不引入 `interrupt()`。
- `SUBMITTED`：仅在确认后创建 `RefundApplication(PENDING)` 和关联审核记录。

管理员审批后的退款状态由现有退款安全流程处理；ConversationSession 只同步展示和下一步提示，不直接改写退款业务状态。

## 6. 工具治理

### 6.1 ToolCapability Registry

保留 LangChain `@tool` 的兼容性，在其外部建立独立注册表和统一 `ToolGuard`，不向框架装饰器塞非标准参数。

```python
@dataclass(frozen=True)
class ToolCapability:
    name: str                                          # 工具名（Registry 内唯一）
    domain: str                                        # 所属业务领域：ORDER / REFUND / POLICY
    allowed_stages: frozenset[str]                     # 允许的工作流阶段；含 "*" 表示全阶段通配（只读工具用）
    required_slots: frozenset[str] = frozenset()       # 调用前必须存在的槽位 / state 字段
    required_eligibility: str | None = None            # last_tool_result.eligibility_passed 必须为 True 的语义标签
    requires_user_confirmation: bool = False           # 是否需要 state.user_confirmed
    idempotency_key_template: str | None = None        # 写工具的幂等键模板（仅 writes_to_conversation=True 时允许）
    writes_to_conversation: bool = False               # 是否会修改会话侧状态（注册时强制：写工具审计钩子必须开启）
    audit_level: str = "read"                          # "read" | "sensitive"；sensitive 必须配 writes_to_conversation=True
    owner_workflow: str = ""                           # 所属工作流（用于审计追溯与 admin 展示）
```

调用路径（`ToolCapabilityRegistry.invoke`，已落地顺序）：

```text
Workflow / ToolNode 选择工具
  → Guard 按序校验：
       1. 域（active_domain / intent 与 capability.domain 一致）
       2. 阶段（"*" 通配 OR stage ∈ allowed_stages）
       3. 槽位（state 与 collected_slots 同时检查）
       4. 资格（last_tool_result.eligibility_passed is True）
       5. 用户确认（state.user_confirmed）
       6. 幂等键生成 + 进程内缓存命中
  → 执行既有 Tool / Service
  → handler 异常被边界吞掉，映射为 ToolOutcome(ok=False, code=SYSTEM_ERROR, retryable=True)
  → Registry 在 envelope 阶段自动填充 9 项元数据（tool_name / domain / workflow_stage /
    conversation_id / thread_id / user_id / order_id / timestamp / idempotency_key）
  → 写工具 + sensitive 审计触发 write_tool_audit_log：
       ok    → AuditAction.PENDING
       !ok   → AuditAction.REJECT
  → 同一 idempotency_key 第二次进入 Registry 时直接复用上次成功 outcome（仅缓存 ok=True）
```

`ToolOutcome` 字段：

| 组 | 字段 | 来源 |
| --- | --- | --- |
| 业务 | `ok` / `code` / `message` / `data` / `retryable` / `idempotency_key` | handler / 业务层 |
| 审计元数据 | `tool_name` / `domain` / `workflow_stage` / `conversation_id` / `thread_id` / `user_id` / `order_id` / `timestamp` | Registry 自动 envelope |

`ToolCode` 枚举（`app/graph/tool_registry.py`）：`SUCCESS` / `ELIGIBILITY_PASSED` / `ELIGIBILITY_REJECTED` / `REFUND_SUBMITTED` / `ALREADY_EXISTS` / `MISSING_SLOT` / `NOT_AUTHORIZED` / `INVALID_STAGE` / `NOT_CONFIRMED` / `BUSINESS_REJECTED` / `TEMPORARY_FAILURE` / `SYSTEM_ERROR`。

#### P2 已登记能力清单（落地状态）

| 工具名 | domain | allowed_stages | 关键约束 | 审计 |
| --- | --- | --- | --- | --- |
| `check_refund_eligibility` | REFUND | `{ELIGIBILITY_CHECKED}` | 需 `active_order_sn` + `user_id` | read |
| `submit_refund_application` | REFUND | `{WAITING_CONFIRMATION, SUBMITTED}` | 需 `active_order_id/sn` + `refund_reason` + `refund_reason_category`，资格通过，用户确认；幂等键 `refund:{user_id}:{order_id}` | sensitive |
| `query_refund_status` | REFUND | 全 8 个退款阶段 | 需 `user_id` | read |
| `query_order_tool` | ORDER | `{"*"}`（全阶段通配） | 需 `user_id` | read |

#### 三道防线（注册校验 + Guard + 业务层）

1. **注册校验**：`register()` 强制 `allowed_stages` 非空、`idempotency_key_template` 仅写工具可用、`audit_level=sensitive` 必须配 `writes_to_conversation=True`；重复注册直接抛错。
2. **Guard 拒绝**：7 步检查任一失败直接返回结构化 `ToolOutcome(code=...)`，handler 不执行；FSM / ToolNode 两条入口共享同一拒因。
3. **业务层**：`RefundApplication.order_id` 物理唯一约束 + Registry 进程内 `_idempotency_cache`（先于 DB 命中）双层短路，并发提交由 `IntegrityError` 兜底。

示例：`submit_refund_application` 必须要求已确认订单、退款原因、资格通过、`WAITING_CONFIRMATION` 阶段和用户确认；任何模型误调用都由 ToolGuard 拒绝。

`GuardedToolExecutor` 是 Registry 的薄封装（`app/graph/tool_registry.py`），是 workflow 节点与 LangChain `@tool` 薄壳的**唯一入口**，避免出现绕过 Guard 的裸 ToolNode 路径。

### 6.2 自然语言输出

不新增全局 Composer 节点。每个 workflow 仅基于本 workflow 已验证的结构化字段生成结果话术：

- PolicyWorkflow 继续使用现有 RAG + 引用 Guardrail；
- OrderWorkflow 根据订单工具结果生成话术；
- RefundWorkflow 根据阶段、槽位和工具 outcome 生成话术。

只有未来出现跨领域结果聚合时，才评估独立 Aggregation Composer，避免二次 LLM 改写带来的延迟、成本和未授权承诺。

## 7. 模块职责与目录建议

```text
app/
  conversation/
    models.py                 ConversationSession ORM
    state_manager.py          working memory 合并与摘要压缩
    transition_resolver.py    domain 连续/切换判断
  graph/
    orchestrator.py           Thin Orchestrator
    tool_registry.py          ToolCapability / ToolGuard / ToolOutcome / GuardedToolExecutor / 审计钩子
    tools.py                  core_* handler + LangChain @tool 薄壳（薄壳只路由到 Guard）
    workflows/
      refund.py               RefundWorkflow 子图（FSM + 工具登记 + 状态更新）
      order.py                OrderWorkflow 子图（P3 已落地：query_order → 自合成 answer，order_data 字段映射 order_id→id）
  services/
    refund_service.py         保留业务规则与数据库操作
    policy_*                  保持现有政策检索与 Guardrail
```

现有 `app/graph/nodes.py` 中的意图路由、订单查询、政策回答和退款 Agent 应按阶段逐步迁移，避免一次性重写。

`app/graph/tool_registry.py` 是 P2 落地的关键模块：ToolCapability / ToolGuard / ToolOutcome / GuardedToolExecutor / write_tool_audit_log 五件事统一在此维护；workflow 与 `@tool` 薄壳都通过 `GuardedToolExecutor.invoke(...)` 接入，禁止出现绕过 Registry 的并行路径。

## 8. 实施阶段与验收

### P0：连续性基础

- 新增 `ConversationSession`、稳定 `conversation_id` 和服务端映射；
- 扩展 `AgentState` Working Memory；
- 实现槽位提取、合并规则、摘要压缩与 `active_domain` 管理；
- 为前端/API 传递并复用 `conversation_id`。

验收：用户先提出退款、后续仅补订单号或原因时，系统不重复询问已确认字段；刷新前端后仍能恢复活跃会话。

### P1：RefundWorkflow FSM

- 将退款自由 Tool loop 改为显式阶段子图；
- 实现订单识别、原因收集、资格校验、确认、提交和状态查询；
- 将对话阶段与 RefundApplication/AuditLog 真源明确分离。

验收：同一退款流程中每个槽位最多主动询问一次；未完成确认不能提交；已提交申请不会再次创建。

### P2：工具能力治理

- 建立 ToolCapability Registry、ToolGuard 和结构化 ToolOutcome；
- 为所有退款写操作和核心订单工具登记阶段、槽位、审计和幂等约束；
- 在管理员退款队列显示 conversation summary、关键槽位和最近工具结果。

验收（按 P2 收尾后实测落地）：

- **域 / 阶段 / 槽位 / 资格 / 确认拒绝路径**：模型在错误阶段、错误领域、缺槽位、资格未通过或未确认时调用工具时被确定性拒绝，handler 不执行，FSM 与顶层 ToolNode 共享同一拒因（`ToolCode` 枚举 + `ToolOutcome` 透传）。
- **元数据自动 envelope**：Registry 在 invoke 阶段用 `dataclasses.replace` 填充 `tool_name / domain / workflow_stage / conversation_id / thread_id / user_id / order_id / timestamp`，handler 只返回业务结果，FSM / admin / 审计读同一结构化字段。
- **审计独立通道**：sensitive 工具（`submit_refund_application`）每次 outcome 都通过 `write_tool_audit_log` 写入 `AuditLog.decision_metadata + context_snapshot`，`AuditAction.PENDING`（成功）/ `AuditAction.REJECT`（失败）；写入失败不回滚业务结果。
- **进程内幂等命中**：`submit_refund_application` 的 `idempotency_key_template="refund:{user_id}:{order_id}"` 在同一进程重复调用时直接复用上次成功 outcome（仅缓存 `ok=True`），与下层 `RefundApplication` 唯一约束形成双层短路。
- **read-only 通配工具**：`query_order_tool`（domain=ORDER）以 `allowed_stages={"*"}` 在所有退款阶段可调用，跨域只读场景不被阶段校验误伤。
- **管理员审批无需重问**：`AuditTask` 新增 `refund_amount / refund_risk_level / refund_status / workflow_stage / user_confirmed / last_tool_outcome`，数据从 `last_tool_result.tool_outcome.data` 与 `context_snapshot.refund` 双源读取；列表查询用 `WHERE thread_id IN (...)` 批量加载 `ConversationSession`，消除 N+1。
- **handler 异常可控**：handler 抛异常时 Guard 边界包装为 `ToolOutcome(ok=False, code=SYSTEM_ERROR, retryable=True, data={"error_type": ...})`，不向上抛，FSM 可按 `retryable` 决定是否回退重试。

测试位于 `test/test_tool_registry.py`（Registry / Guard / 元数据 envelope / 顶层 ToolNode / 管理员字段 28 用例），与 `test_refund_workflow_fsm.py` / `test_refund_tools.py` / `test_conversation_history.py` 共 54 用例全过；改动文件 ruff 干净。

### P3：领域工作流拆分

- 将 OrderWorkflow 拆为子图（`app/graph/workflows/order.py`）；
- PolicyWorkflow 保持现有单节点，不做无意义拆分；
- 完成 Thin Orchestrator 接入与回归测试；
- `_register_capabilities` 改为 per-name 幂等注册（refund 与 order 两模块互不阻断）；
- 子图模块级编译一次复用（`_ORDER_SUBGRAPH` / `_REFUND_SUBGRAPH`）；
- 顶层 `ToolNode(refund_tools+order_tools)` 死代码移除，工具薄壳保留供 LangChain 兼容与测试；
- `query_order_tool` 归属从 `OrderQuery` 改为 `OrderWorkflow`；
- 主图新增 `dispatch_router`：按 `active_domain` 派发到 `order_workflow` / `retrieve` / `refund_agent`，无活跃域才走 `intent_router`；
- SSE 兜底事件捕获集合加入 `order_workflow`，确保 ORDER 确定性合成的 answer 经 SSE 输出。

验收（已实测落地）：

- **ORDER/POLICY/REFUND 切换不丢旧流程摘要**：`ConversationStateManager.persist_turn` 与 `intent`/`active_domain` 同步，summary 在跨域切换时被保留。
- **ORDER→REFUND 连续会话不丢 `active_order_id`**：OrderWorkflow 子图把 `core_query_order` 返回的 `outcome.data["order_id"]` 映射为 `order_data["id"]`，供 `persist_turn` 固化（`state_manager.py:288` 读 `order_data.get("id")`）。
- **SSE 输出**：捕获集合 `{"generate", "refund_agent", "order_workflow"}`，无 LLM 流的 ORDER 路径通过 fallback answer 输出。
- **per-name 幂等注册**：重复调用 `_register_capabilities` 不抛错。
- **Policy Guardrail / 退款审批 / 支付安全回归**：`test_tool_registry.py` + `test_refund_workflow_fsm.py` + `test_refund_tools.py` + `test_conversation_history.py` + 新增 `test_order_workflow.py` 共 64 用例全过；改动文件 ruff 干净。

## 9. 暂缓事项

- 通用 Case/Ticket、SLA、坐席 owner 和跨渠道人工实时接管；
- Supervisor 自由调度多个 Agent；
- 通用 Plan-and-Execute；
- LangGraph `interrupt()`；
- 跨领域聚合 Composer；
- 长期用户画像与偏好记忆；
- 真实支付的支付流水、网关幂等键和 provider transaction ID。

这些能力应在 ConversationSession、RefundWorkflow 和 ToolGuard 稳定后按实际复杂度再引入。
