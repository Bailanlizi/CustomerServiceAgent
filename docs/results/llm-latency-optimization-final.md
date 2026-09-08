# LLM 延迟优化最终报告

> 状态：已完成（P0–P2 四步迭代 + 7 轮全量终验）。  
> 基线：`eval/runs/agent_e2e_20260907_084355.json`（8 场景 × 7 轮，`qwen3.7-max` 全局单模型）  
> 终验：`eval/runs/agent_e2e_20260907_140529.json`（8 场景 × 7 轮，`qwen3.7-flash`/`max` 分层 + `enable_thinking: false`）  
> 配套设计文档：`docs/llm-latency-optimization.md`（每步落地点与踩坑更正）

## 一、优化目标

针对 8 场景端到端评估中暴露的延迟瓶颈（`llm_only` p50=9.6s / p95=15.7s），在**不改变业务正确性与安全口径**的前提下做四步迭代：

1. 退款三话术节点用确定性模板替代 LLM 流式调用
2. 高频轻任务切到 `qwen3.7-flash` + 关闭 thinking；extractor prompt 做记忆白名单裁剪
3. 精简三处 prompt + 政策回答路径关闭 thinking
4. 政策回答校验后做 SSE 切片推送（体验改善）

## 二、关键指标：基线 vs 终验（同 7 轮口径）

| 指标 | 基线 (084355) | 终验 (140529) | 变化 |
| --- | --- | --- | --- |
| 正常业务成功率 | 1.0 (49/49) | 1.0 (49/49) | 持平 |
| 间接安全信号率 | 1.0 (7/7) | 1.0 (7/7) | 持平 |
| 工具选择 Precision / Recall | 1.0 / 1.0 | 1.0 / 1.0 | 持平 |
| 工具顺序正确率 | 1.0 (n=42) | 1.0 (n=42) | 持平 |
| valid_runs / 场景 | 全 7 | 全 7 | 持平 |
| degenerate_runs | 全 0 | 全 0 | 持平 |
| **`llm_only` p50** | 9617ms | **910ms** | **-91%** |
| **`llm_only` p95** | 15677ms | **2405ms** | **-85%** |
| LLM 调用总数 | 84 | 77 | -7（步骤 1 消灭 7 次退款话术流式） |
| prompt tokens 总值 | 33814 | 22834 | -33% |
| **completion tokens 总值** | 40533 | **5493** | **-86%** |
| exact_invoke / exact_stream | 77 / 7 | 77 / 0 | 步骤 1 后无流式调用 |

## 三、节点延迟对比

| 节点 | 基线 p50 / p95 | 终验 p50 / p95 | 说明 |
| --- | --- | --- | --- |
| `generate`（政策生成） | 15679 / 23001 | **2408 / 2643** | 步骤 3 关 thinking 后大幅下降 |
| `identify_order` | 4726 / 6404 | **0.6 / 1.4** | 步骤 1 改为确定性模板 |
| `refund_agent` | 20 / 5965 | **20 / 50** | 步骤 1 消灭 LLM 话术 |
| `retrieve` | 788 / — | 498 / 598 | retrieve + load_policy 并行，主要非 LLM |
| `entry` / `dispatch_router` / `route_from_entry` / `compose_answer` | <30 / <135 | <3 / <3 | 业务逻辑层零变化 |
| `check_eligibility` / `submit` / `query_order` / `order_workflow` | <30 / <135 | <33 / <150 | 业务逻辑层零变化 |

## 四、四步迭代摘要

### 步骤 1（P0）：退款三话术节点确定性模板化

**文件**：`app/graph/workflows/refund.py`

三个原本用 `astream` 生成追问话术的节点（`node_identify_order` / `node_collect_reason` / `node_await_confirmation`）全部改为固定模板字符串插值。这三个节点的输出本质是固定话术 + 状态字段，且 extractor 在 prepare_turn 阶段已预填订单号/原因，FSM 每轮只走一个阶段、工具结果自带确定性文案，三个 LLM 话术节点在 eval 56 个 run 里一次都没触发——属于用 LLM 做本质确定性的工作。

**验证**（`agent_e2e_20260907_105130.json`，`--runs 2`）：16/16 全通过。`identify_order` p50 4726→1.1ms，`refund_agent` p95 5965→225ms，`exact_stream` 7→0。

### 步骤 2（P0）：模型分层 + extractor prompt 裁剪

**文件**：`app/core/config.py` / `.env` / `app/graph/nodes.py` / `app/conversation/state_manager.py`

- 新增 `LLM_MODEL_FAST=qwen3.7-flash` 配置项
- `intent_classifier`、闲聊 `generate`、`extractor`、`summarizer` 切到 flash；`policy_answer_llm` 保留 max
- extractor prompt 做记忆白名单投影：只传 `active_domain / active_order_sn / conversation_summary / pending_slots / collected_slots[白名单子集]`，**不传** `last_tool_result / workflow_stage / next_action / active_order_id`
- 指令文本压缩至 ~100 字，枚举值完整保留

**首验踩坑**（已记入设计文档）：首验 16 runs 中 S03/S07/S08 全挂。根因：`qwen3.7-flash` 默认开启 thinking，结构化调用输出 550-900+ tokens 推理文本，把 `max_tokens` 预算烧光 → JSON 截断异常 → 静默走关键词兜底（"我要退 SN..." 不含"退款/退货" → 误判 OTHER → FSM 停在 collect_reason）。

**修复**：快模型统一加 `extra_body={"enable_thinking": False}`。实测 output 550→111 tokens，单次延迟 3.4s→0.8s。**教训：换快模型必须先单独直连测 structured output 的真实 completion 开销，再定 max_tokens。**

**修复后验证**（`agent_e2e_20260907_131155.json`，`--runs 2`）：16/16 全通过。`llm_only` p50 9078→894ms，`intent_router` LLM 触发恢复为 0，completion 总量 11039→3211，prompt 总量 9520→6564。

### 步骤 3（P1）：Prompt 清理与政策回答限长

**文件**：`app/graph/nodes.py`

- 删除死代码 `PROMPT_TEMPLATE` + `prompt` + `ChatPromptTemplate` import（无引用）
- `INTENT_PROMPT` 删 8 个 few-shot 示例与"只返回标签"冗余指令，保留 4 类定义
- `GENERATE_SYSTEM_PROMPT` 4 条规则合并精简为列表式
- `POLICY_GENERATE_SYSTEM_PROMPT` 5 条规则合并为 4 条，加「answer ≤120 字」软约束
- 政策路径单独构造 `_policy_llm` 实例：保留 max 模型（保证 PolicyAnswerGuard 引用合规）+ `enable_thinking: False`

**实施更正**（已记入设计文档）：方案原本计划加 `max_tokens=700` 硬限，**未加**——结构化 JSON 输出截断会触发 PolicyAnswerGuard 校验失败 → 重试延迟翻倍（方案预警的风险）。直连实测 max 模型 thinking 开/关：单条 9.8s→2.7s，引用合规不变（CAT_001/[CAT_001,FAQ_011]）。**这是 p95 下降的核心。**

**验证**（`agent_e2e_20260907_132841.json`，`--runs 2`）：16/16 全通过。`llm_only` p95 15768→2319ms，`generate` 节点 p50 16924→2614ms，S05 completion 1023→103，`retry_count=0`。

### 步骤 4（P2）：政策回答校验后分段推送

**文件**：`app/api/v1/chat.py` / `test/test_chat_sse.py`

`fallback_answer` 从整段一次性 yield 改为按 20 字切片逐片 yield，片间 `await asyncio.sleep(0.02)`。SSE 事件格式与流式路径一致，前端按 token 拼接逻辑已存在。

**设计约束**（与方案一致）：**真流式不做**。`chat.py:145-146` 丢弃 `POLICY_GUARD_TAG` 流式 token 是有意安全设计（未通过引用校验的 JSON 片段不能泄露给前端），只做校验后分段推送，总延迟不变、纯 TTFT 体验改善。

SSE 单测断言从「整段 token 事件 in body」改为「所有 token 事件拼接 == 完整答案」。

**验证**（`agent_e2e_20260907_135254.json`，`--runs 2`）：16/16 全通过，单测 67 全过（含 SSE 3 个）。延迟持平符合预期。

## 五、7 轮全量终验（`agent_e2e_20260907_140529.json`）

8 场景 × 7 轮 = 56 runs，与基线同口径。

- **56/56 全通过**（S01–S08 各 7/7）
- 工具 precision/recall/order 全 1.0，零退化 run
- `llm_only` p50 = **910ms**（基线 9617ms，-91%），p95 = **2405ms**（基线 15677ms，-85%）
- LLM 调用 77 次（基线 84，步骤 1 消灭 7 次流式话术）
- completion tokens 总值 5493（基线 40533，-86%）
- `generate` 节点 p50 = 2408ms（基线 15679ms，-85%），p95 = 2643ms（基线 23001ms，-89%）
- S06 越权仍走 indirect 验证（`security_intercept_rate=null`），与基线口径一致

## 六、模型分层与 thinking 配置

终态模型实例化（`app/graph/nodes.py` + `app/conversation/state_manager.py`）：

| 实例 | 模型 | thinking | 用途 |
| --- | --- | --- | --- |
| `llm` | `qwen3.7-max` | 默认（开） | 兜底主模型，当前无直接调用方 |
| `fast_llm` | `qwen3.7-flash` | **关** | 意图分类、闲聊 generate |
| `_policy_llm` | `qwen3.7-max` | **关** | 政策回答（保证引用合规） |
| extractor | `qwen3.7-flash` | **关** | 槽位抽取（高频，每轮调用） |
| summarizer | `qwen3.7-flash` | **关** | 会话压缩（≥6 轮触发） |

**回退**：`.env` 把 `LLM_MODEL_FAST` 改回 `qwen3.7-max` 即可整体回退到优化前配置。

## 七、关键经验教训

1. **换快模型必须先单独直连测 structured output 的真实 completion 开销**：qwen3.7-flash 默认 thinking 输出 550-900+ tokens 推理文本，远超结构化 schema 本身需要的 ~100 tokens，会顶掉 `max_tokens` 预算导致 JSON 截断。`enable_thinking: false` 是 qwen3 系列快模型用于结构化抽取的必备参数。

2. **结构化输出的 max_tokens 硬限要谨慎**：截断会触发 schema 校验异常 → 静默走兜底 → 难以定位的回归。软约束（prompt 写「≤120 字」）+ 关 thinking 比硬限更稳。

3. **确定性输出不应过 LLM**：退款三话术节点本质是固定文案 + 状态插值，用 LLM 生成既慢又不稳定。模板化后 refund_agent p95 从 5965ms 降到 50ms，话术一致性反而更好。

4. **记忆白名单投影**：extractor 把整个 memory JSON 塞进 prompt 会让第二轮起 prompt 从 240 涨到 730 tokens。只传抽取实际需要的 5 个字段，prompt 稳定在 ~150 tokens。

5. **安全路径不要做真流式**：PolicyAnswerGuard 校验未通过的 token 不能泄露给前端。校验后分段推送是唯一安全且能改善 TTFT 感知的方案。

## 八、后续可选方向（未实施）

- **P3（可选）**：政策规则块显式 prompt 缓存（DashScope `cache_control`，最小 1024 tokens，TTL 5 分钟）。只省 prefill，瓶颈已在 completion，收益最小。
- **政策模型 max→plus A/B**：政策路径仍是 p95 大头（2.4s），若 plus 能保持引用合规可再降 1-2s，需单独跑 S05 7 轮验证。
- **S06 安全测试方法升级**：从 indirect 验证恢复为直接验证 `ToolCapabilityRegistry` 的 `NOT_AUTHORIZED` 拦截路径被实际触发。
