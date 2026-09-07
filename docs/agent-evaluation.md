# Agent 端到端评估方案 — 设计决策

> 状态：**已实施**（2026-09-07，最终产物 `eval/runs/agent_e2e_20260907_041503.json` / `.md`）
>
> 本文档职责单一：定义「Agent 任务成功率、工具调用准确率、单次对话链路 Trace」三套指标的测量口径、场景集、采集方案与输出产物。
>
> **真源边界**：本文档不重复 `architecture-update.md` 的架构设计，不重复 `chat-history-recovery.md` 的会话持久化设计；仅引用它们的结论。落地后，本节评估产生的**最终数字**应沉淀进 `README.md`（或本文档 §9 的产物），设计真源仍在 `architecture-update.md`。

---

## 1. 背景与目标

本项目定位为「客服 AI Agent」，投递 AI 应用 / Agent 开发岗。需要一组**可复现、可解释、面试可展示**的量化指标，证明 Agent 真实工作的能力，而非仅靠测试覆盖率。

本方案测三件事：

| 维度 | 指标 | 回答的问题 |
|---|---|---|
| 任务层 | 任务成功率 | 「Agent 真的把任务跑通了吗？」 |
| 行动层 | 工具调用准确率 | 「该调工具时选对、传对了吗？」 |
| 链路层 | 节点耗时 + LLM Token | 「哪一步慢？哪次模型调用最贵？」 |

**不做**（范围外，理由见 §11）：进度率、PlanQuality/PlanAdherence、步骤效率、Token 成本优化、稳定性 σ、生产级监控、LangSmith 接入。

---

## 2. 范围决策（已定稿）

1. **8 个场景**（非 12 个），覆盖正常任务 / 多轮状态 / RAG / 权限 / 幂等 / 失败路径。
2. **正常业务成功率**与**越权拦截率**分开统计，安全场景不进成功率分母。
3. **工具准确率**只对「实际存在工具调用」的场景统计。
4. 同场景重复 **N=2** 次观察非确定性；结果声明为**小样本工程评估**，非生产置信区间。两次不一致时仅对该场景补跑第 3 次。
5. 输出 **JSON + Markdown** 两份产物。

---

## 3. 代码事实基线（只读核实结论）

方案全部锚定以下已核实的代码事实，避免「计划 vs 实现」偏差。

### 3.1 工具注册名（真源：`app/graph/tools.py`）

| 工具 | 注册名 | 领域 | 审计级别 | 关键 handler |
|---|---|---|---|---|
| 订单查询 | `query_order_tool` | ORDER | read | `core_query_order` (tools.py:543) |
| 退款资格预检 | `check_refund_eligibility` | REFUND | read | `core_check_refund_eligibility` (tools.py:64) |
| 提交退款申请 | `submit_refund_application` | REFUND | **sensitive** | `core_submit_refund_application` (tools.py:166) |
| 退款进度查询 | `query_refund_status` | REFUND | read | — |

> 不存在 `policy_search` 工具——政策问答走 `retrieve` + `generate` 节点，不是工具调用（场景 5 因此不计入工具准确率）。
> `refund.py:21` 注释写 `query_order` 是过时注释，实际注册名是 `query_order_tool`。

### 3.2 两套枚举（勿混淆）

| 枚举 | 取值 | 含义 | 位置 |
|---|---|---|---|
| `RefundStatus`（DB 状态） | PENDING / APPROVED / PROCESSING / REJECTED / COMPLETED / CANCELLED | 退款申请记录状态，**无 SUBMITTED** | `app/models/refund.py:9` |
| `RefundStage`（FSM 阶段） | IDLE → IDENTIFY_ORDER → COLLECT_REASON → ELIGIBILITY_CHECKED → WAITING_CONFIRMATION → SUBMITTED → DONE / REJECTED | 退款子图状态机，**含 SUBMITTED** | `app/graph/workflows/refund.py:50` |

→ 退款刚提交时，**DB 记录状态是 `PENDING`**（待人工审），不是 `SUBMITTED`。

### 3.3 越权拦截判定源

- `AuditAction` = APPROVE / REJECT / ESCALATE / PENDING（`app/models/audit.py:21`），**无 PERMISSION_DENIED**。
- 越权拒绝的判定源是 `ToolOutcome.code == NOT_AUTHORIZED`（`app/graph/tool_registry.py:37`）。
- 越权落点：`core_query_order` / `core_check_refund_eligibility` / `core_submit_refund_application` 均用 `Order.user_id == user_id` 过滤，查不到返回 `NOT_AUTHORIZED`（tools.py:575 / 215 / 220）。
- **read 工具不写 audit_log**（`audit_level="read"`）；只有 sensitive 的 `submit_refund_application` 每次 outcome 写 `audit_logs`（`write_tool_audit_log`, tool_registry.py:340）。因此「审计有无」取决于走了哪条路径，**不稳定**，不作为安全成功条件。

### 3.4 图节点名（真源：`workflow.py` / `workflows/*.py`）

| 层级 | 节点名 |
|---|---|
| 主图 | `order_workflow`(子图) / `retrieve` / `refund_agent` / `generate` / `intent_router`；入口条件边 `dispatch_router` |
| order 子图 | `query_order` → `compose_answer` |
| refund 子图（6 阶段） | `entry` / `identify_order` / `collect_reason` / `check_eligibility` / `await_confirmation` / `submit` |

> `load_session` / `memory_rehydrate` 是 chat.py 会话层步骤（`prepare_turn`, state_manager.py:169），**不是 Graph 节点**，Trace 中记为 `pre_graph` 事件。

### 3.5 工具执行钩子点

- 3 个 `GuardedToolExecutor` 实例（tools.py:45 / order.py:30 / refund.py:64），但共享全局单例 `tool_registry`。
- **唯一 hook 点 = `ToolCapabilityRegistry.invoke`（tool_registry.py:139）**，一处覆盖全部 4 个工具 + 所有调用路径（LangChain 薄壳 + workflow 直接调用）。
- hook 时可一次性拿到：`name`（工具名）+ `**arguments`（业务参数）+ `state`（注入参数）+ 返回的 `ToolOutcome`（含 `tool_name/code/ok/user_id/order_id/timestamp` 等 envelope 已填充字段）。

### 3.6 LLM 实例与 purpose

- 全项目 **2 个 ChatOpenAI 实例**：`nodes.py:32`（`llm`，用于意图分类/政策生成/通用话术/退款话术）+ `state_manager.py:107`（`base_llm`，用于抽槽 + 摘要）。
- refund 子图有 **3 处 LLM**（refund.py:146/175/225，复用 `nodes.llm`），order 子图**无 LLM**（compose_answer 为纯格式化）。
- 6 个 purpose：`intent_classification` / `policy_generation` / `generic_generation` / `refund_prompt` / `memory_extraction` / `summarization`。
- 因 refund.py 复用 `nodes.llm` 裸实例未打 tag，区分 purpose 需在各调用点 `with_config({"tags":[...]})` 打 tag，由 callback 读 tags。

### 3.7 seed 数据（`scripts/seed_data.py`）

| 用户 | 订单 | 状态 | 可否退款 |
|---|---|---|---|
| alice | SN20240001 | SHIPPED | ✅（ALLOWED 含 SHIPPED） |
| alice | SN20240002 | PENDING | ❌ |
| alice | SN20240003 | SHIPPED | ✅ |
| bob | SN20240004 | DELIVERED | ✅ |
| bob | SN20240005 | PAID | ❌ |

- 退款资格规则（`refund_service.py:22-25`）：`ALLOWED_ORDER_STATUSES = {DELIVERED, SHIPPED}`，纯 Python 硬逻辑、不依赖 LLM。
- seed **不含 RefundApplication**，8 场景中的退款场景需评测脚本自行建立前置（或复用场景 3 的结果作场景 8 前置）。
- 无 `SN001`，场景里的订单号必须用 `SN20240001` 等真实号。

---

## 4. 场景集（8 个，最终口径）

| # | 场景 | 输入（alice，除注明） | 允许工具链 | 验收断言 |
|---|---|---|---|---|
| 1 | 查我的订单 | 「查我的订单」 | `query_order_tool` | 返回 alice 订单，不跨用户 |
| 2 | 查指定订单 | 「SN20240001 啥情况」 | `query_order_tool` | 返回 SN20240001 详情 |
| 3 | 正常退款全流程 | 「退 SN20240001 质量问题」→「确认」 | 路径A `check_refund_eligibility`→`submit_refund_application`；路径B `query_order_tool`→`check_refund_eligibility`→`submit_refund_application` | `refund_applications.status == PENDING` |
| 4 | 缺订单号 | 「我要退款」 | 不调工具（反问） | 正确反问要订单号 |
| 5 | 政策问答 | 「内衣试穿能退吗」 | 走 `retrieve`+`generate`（非工具） | 答案命中正确条款（RAG 维度，不计工具准确率） |
| 6 | 越权退款 | alice「退 SN20240004」（bob 的） | 拦截 | 业务：无新增退款申请；工具：`code == NOT_AUTHORIZED`（**不以「无 audit_log」为条件**） |
| 7 | 重登续退款 | 重登后续聊退款 | 续 路径A/B | 同一 conversation，状态恢复 |
| 8 | 重复提交 | 同一订单再次提交 | `submit_refund_application`×2 | DB 只有一条退款申请 + 用户收到「已有申请」提示（**不锁具体 ToolCode**） |

**场景 8 的 code 不稳定说明**：第二次提交若命中 registry 幂等缓存返回 `REFUND_SUBMITTED`（tool_registry.py:238-249），若缓存未命中走 DB 层返回 `ALREADY_EXISTS`（tools.py:326）。两者都等价于「不新增第二条」，因此验收只锁 `DB 只有一条申请` 这一稳定事实。

---

## 5. 指标定义

### 5.1 任务成功率（头部指标）

- **正常业务成功率** = 场景 {1,2,3,4,5,7,8} 中「满足验收断言」的场景数 / 7。
- **越权拦截率** = 场景 {6} 单独一行（`1/1`），不进成功率分母。
- 「成功」= 满足该场景**可机器判定的验收断言**（DB 查询 / `ToolOutcome.code` / 结构化返回），**不依赖 LLM 自评**。

### 5.2 工具调用准确率（只统计有工具调用的场景 1/2/3/6/7/8）

拆 4 维，不要求「完全相等」：

| 维度 | 定义 |
|---|---|
| 选择 Precision | 实际多调了几个**不该调**的工具 |
| 选择 Recall | 期望工具**实际调了几个** |
| 顺序正确率 | 关键依赖顺序是否正确（如 `check_refund_eligibility` 必须先于 `submit_refund_application`） |
| 参数准确率 | 见 5.3 分层 |

> 允许路径采用「允许链」而非唯一链（场景 3 的路径 A/B），`query_order_tool` 作为合理前置不算错误调用。

### 5.3 参数准确率分层

| 层 | 判定 | 说明 |
|---|---|---|
| schema 正确率 | 字段齐全、类型正确 | — |
| 关键值准确率 | 订单号、退款原因 | 模型真正提取的值 |
| **安全参数** | `user_id`/`order_id` 来自认证上下文 | **不算模型传参**——它们是 `InjectedState` 注入（tools.py:341-347），模型只真正传 `reason` |
| 非关键文本 | 允许轻微措辞差异 | — |

### 5.4 链路 Trace 指标

- 节点耗时 p50 / p95（按 event_type 分组）。
- 每次会话 LLM token：`prompt` / `completion` / `by_purpose` 分布。
- embedding 调用次数单独记录，不与对话 LLM token 混算。

---

## 6. Trace 事件模型

```
event_type ∈ pre_graph / main_graph / subgraph_node / tool_call / llm_call
```

| event_type | 内容 |
|---|---|
| `pre_graph` | `load_session`、`memory_rehydrate`（会话层，不伪装成 graph 节点） |
| `main_graph` | `dispatch_router`、`intent_router`、`retrieve`、`generate` |
| `subgraph_node` | `order.query_order`、`order.compose_answer`、`refund.entry/identify_order/collect_reason/check_eligibility/await_confirmation/submit` |
| `tool_call` | 4 个工具的真实调用 |
| `llm_call` | 带 `purpose` 标签的模型调用 |

单条事件字段（采纳审查建议）：

```json
{
  "run_id": "...", "scenario_id": "S03", "turn": 2,
  "node_type": "subgraph_node", "event_type": "tool_call",
  "node": "refund.submit", "tool_name": "submit_refund_application",
  "duration_ms": 420, "status": "success", "error_type": null,
  "llm_usage": {"prompt_tokens": null, "completion_tokens": null, "available": false},
  "model": "gpt-4o", "trace_id": "..."
}
```

**Token 结构**（LLM 与 embedding 分开，缺失不伪造为 0）：

```json
{
  "llm_tokens": {"prompt": 1200, "completion": 300, "by_purpose": {
    "intent_classification": {}, "policy_generation": {},
    "generic_generation": {}, "refund_prompt": {},
    "memory_extraction": {}, "summarization": {}
  }},
  "embedding_calls": 1, "embedding_usage_available": false
}
```

---

## 7. 采集方案（零侵入生产代码）

| 采集对象 | 方式 |
|---|---|
| 实际工具链 + 参数 + 结果 | monkey-patch `tool_registry.invoke`，记录 `name` / `arguments` / `state` / `ToolOutcome` |
| LLM token + purpose | LangChain callback 挂到 2 个 ChatOpenAI 实例；purpose 靠调用点 `tags` 区分 |
| 节点耗时 | LangGraph run 回调读节点级 run（callback 拿不到节点耗时，需 graph 打点）+ 工具外层计时 |
| 任务结果 | DB 断言（`refund_applications` / `audit_logs` / `orders`）+ 结构化 `ToolOutcome.code` |

**执行入口**：复用 `state_manager` 的 `resolve_session` → `prepare_turn` → `graph.ainvoke` → `persist_turn`（`chat.py` `event_generator` 的非流式等价），使场景 7（重登恢复）真实走通。

---

## 8. 输出产物

| 文件 | 用途 |
|---|---|
| `eval/runs/agent_e2e_<timestamp>.json` | 机器复查（全量 per-scenario + trace） |
| `eval/runs/agent_e2e_<timestamp>.md` | 面试展示（汇总 + purpose token 分布表 + 失败分布） |

报告声明：每个场景重复 2 次观察非确定性，结果为小样本工程评估，不代表生产置信区间。

---

## 9. 实施清单（已全部完成）

1. [x] 只读核实完毕（本 §3 已产出）。
2. [x] 写 `scripts/eval_agent_e2e.py`：monkey-patch `tool_registry.invoke` + LLM callback + graph 打点。
3. [x] 8 场景 oracle 用 §3.1 注册名 + §3.4 节点名 + FSM 阶段定义（不「先跑取 actual 当 expected」）。
4. [x] 场景 6 的越权断言聚焦「业务无新增 + `code==NOT_AUTHORIZED`」。
5. [x] 场景 8 的幂等断言聚焦「DB 单条申请」。
6. [x] 跑 8 场景 × N=2，输出 §8 两份产物。
7. [x] 把最终数字沉淀进 `README.md`，本文档归档为「已实施」。

---

## 10. 风险与边界

| 项 | 状态 | 影响 |
|---|---|---|
| 「工具调用准确率」业界无统一基准 | 用「oracle 期望链」作项目自定口径 | 面试需主动说明 |
| LLM 非确定性 | N=2 小样本 | 报告已声明非置信区间 |
| 退款流程依赖支付/配额等 | 现有 seed 仅订单，退款申请靠脚本自建前置 | 场景 3→8 需顺序编排 |
| 场景 6 越权路径 | 取决于模型是否先走 `query_order_tool` | 断言只看业务结果 + `NOT_AUTHORIZED`，不锁路径 |
| token 网关可能不返回 usage | 允许 null | 已声明 `usage_available` |

---

## 11. 范围外（不做）

进度率、PlanQuality / PlanAdherence（本项目无 planner，FSM 驱动）、步骤效率、Token 成本优化、稳定性 σ（10 次重跑）、生产级监控（Prometheus/Grafana/SLO）、LangSmith 完整接入、多轮基准（MT-Bench / AlpacaEval）。

---

## 12. 落地结果（2026-09-07）

最终产物：`eval/runs/agent_e2e_20260907_041503.json` / `.md`（`qwen3.7-max`，8 场景 × N=5）。

### 12.1 核心指标

| 指标 | 值 | 口径 |
|---|---|---|
| 正常业务成功率 | **1.0**（35/35） | 7 个正常场景 × 5 次，全过 |
| 越权拦截率 | **1.0**（5/5） | 场景 6 独立统计，不进成功率分母 |
| 工具选择 Recall | **1.0** | 期望工具全部被调用 |
| 工具选择 Precision | **0.857** | 7 次调用中 6 次命中允许链 |
| 工具顺序正确率 | **0.833** | 6 个有工具场景中 5 个顺序正确 |

### 12.2 链路 Trace（p50 节点耗时，单位 ms）

| 节点 | p50 | 说明 |
|---|---|---|
| `identify_order` | 3695.7 | 退款子图内 LLM 抽取订单号（最贵单点） |
| `generate` | 14987.8 | 政策生成 + 引用校验（S05 单场景） |
| `retrieve` | 863.5 | 向量检索 |
| `check_eligibility` / `submit` / `query_order` | < 40 | 确定性业务逻辑，无 LLM |

- LLM 调用次数：60；prompt 23,787 tokens；completion 27,368 tokens。

### 12.3 口径与边界（面试需主动说明）

1. **Precision 0.857 与 order 0.833 的失分点全部来自 S08（幂等场景）**。该场景 `allowed_chains=[]`（预期不产生新申请），但 Agent 仍调用 `check_refund_eligibility` 去探测是否已有申请——这是合理的防御性查询，DB 层面也确实未新增申请。按「严格口径」记为一次额外调用与一次顺序偏差；从业务正确性看 S08 完全正确。
2. **Token 统计只覆盖 `ainvoke` 路径**。退款子图节点内直调 `llm.astream`（流式），网关不返回 `usage_metadata`，故这些调用 token 记为 `null`，未计入总量。`identify_order` 等节点的耗时已计入，仅 token 缺失。
3. **N=5 为小样本工程评估**，非生产置信区间；LLM 非确定性下不建议对外宣称 100%。

### 12.4 实施要点（调试沉淀）

- 场景 3/7/8 存在**跨 run 数据污染**：`submit_refund_application` 走进程内幂等缓存（`tool_registry._idempotency_cache`，key `refund:{user_id}:{order_id}`）。仅删 DB 行不删缓存，会导致 N≥2 时 submit 命中缓存返回 `REFUND_SUBMITTED` 却不再落库。故 `cleanup_order_refund` 在删 DB 前先 `tool_registry._idempotency_cache.clear()`。
- 越权场景（S06）与重登退款（S07）需用**不同订单号**隔离（SN20240005 vs SN20240004），否则跨场景互相污染。
