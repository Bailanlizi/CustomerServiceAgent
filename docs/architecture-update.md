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
class ToolCapability:
    domain: str
    requires_stages: set[str]
    requires_slots: set[str]
    requires_eligibility: str | None
    idempotency_key_template: str | None
    writes_to_conversation: bool
    audit_level: Literal["read", "write", "sensitive"]
```

调用路径：

```text
Workflow 选择工具
  → ToolGuard 校验领域、阶段、槽位、资格、权限和幂等条件
  → 执行既有 Tool / Service
  → 返回结构化 ToolOutcome
  → 更新工作记忆、审计和自然语言结果
```

示例：`submit_refund_application` 必须要求已确认订单、退款原因、资格通过、`WAITING_CONFIRMATION` 阶段和用户确认；任何模型误调用都由 ToolGuard 拒绝。

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
    tool_registry.py          ToolCapability 与 ToolGuard
    workflows/
      order.py                OrderWorkflow 子图
      refund.py               RefundWorkflow 子图
  services/
    refund_service.py         保留业务规则与数据库操作
    policy_*                  保持现有政策检索与 Guardrail
```

现有 `app/graph/nodes.py` 中的意图路由、订单查询、政策回答和退款 Agent 应按阶段逐步迁移，避免一次性重写。

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

验收：模型在错误阶段调用工具时被确定性拒绝；管理员审批无需重新询问订单、原因和资格结果。

### P3：领域工作流拆分

- 将 OrderWorkflow 拆为子图；
- PolicyWorkflow 保持现有单节点，不做无意义拆分；
- 完成 Thin Orchestrator 接入与回归测试。

验收：ORDER/POLICY/REFUND 切换不丢失旧流程摘要；Policy Guardrail、退款审批与支付安全回归全部通过。

## 9. 暂缓事项

- 通用 Case/Ticket、SLA、坐席 owner 和跨渠道人工实时接管；
- Supervisor 自由调度多个 Agent；
- 通用 Plan-and-Execute；
- LangGraph `interrupt()`；
- 跨领域聚合 Composer；
- 长期用户画像与偏好记忆；
- 真实支付的支付流水、网关幂等键和 provider transaction ID。

这些能力应在 ConversationSession、RefundWorkflow 和 ToolGuard 稳定后按实际复杂度再引入。
