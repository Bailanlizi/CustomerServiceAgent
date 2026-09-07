# LLM 延迟优化方案 — 分层模型 / Prompt 精简 / 流式体验 / 缓存

> 状态：**待实施**（2026-09-07 定稿）
>
> 本文档职责单一：记录「LLM 调用延迟高」的根因分析、四个优化方向的评估结论、按步骤的落地改动清单，以及**冻结的验证协议**。
>
> **真源边界**：本文档不重复 `architecture-update.md` 的架构设计、`agent-evaluation.md` 的评估口径；仅引用结论。评估产物仍落在 `eval/runs/`，最终数字沉淀进 README。
>
> **基线锚点**：`eval/runs/agent_e2e_20260907_084355.json`（qwen3.7-max，7 轮，8 场景 56/56 全通过）。

---

## 1. 背景与基线

端到端评估功能层已全绿（8 场景 × 7 轮全部通过，工具选择 precision/recall/order 均为 1.0），但延迟完全由 LLM 调用支配：

| 指标（基线 run） | 数值 |
|---|---|
| llm_only_latency p50 / p95 | **9617ms / 15677ms**（n=84） |
| 业务逻辑节点 p95 | 全部 < 135ms（dispatch_router / entry / query_order / submit 等） |
| generate 节点 p50 / p95（政策问答） | 15679ms / 23001ms |
| identify_order 节点 p50 / p95 | 4726ms / 6404ms |
| refund_agent p95 | 5965ms（p50 仅 20ms，慢路径是 LLM 话术节点） |
| LLM 调用构成 | 84 次：exact_invoke 77（非流式）/ exact_stream 7 |
| Token 总量 | prompt 33814 / completion 40533（**平均 completion 482 tokens/次，比 prompt 还多**） |

### 根因（五重叠加）

1. **模型大**：全局单一 `qwen3.7-max`，所有任务共用（`app/graph/nodes.py:32-37`、`app/conversation/state_manager.py:107-114`）。
2. **非流式占 92%**：77/84 次走 `ainvoke()`，首 token 延迟 = 总延迟。
3. **结构化输出走 function calling**：政策回答 `with_structured_output(PolicyAnswer)`，比纯文本慢且 completion 长（S05 单次 completion 达 1023）。
4. **串联多次调用**：每轮平均 1.5 次 LLM 调用；政策问答单轮 2 次（extractor 分类 + 政策生成），退款多轮流程每轮 1 次。
5. **生成 token 过多**：固定话术、分类、抽取这类轻任务也在生成长 completion。

---

## 2. LLM 调用全景（优化前事实清单）

| # | 调用点 | 模式 | eval 频次 | 任务性质 | 处置决策 |
|---|---|---|---|---|---|
| 1 | extractor 槽位抽取 `state_manager.py:321` | structured ainvoke | **每轮，约 70/84** | 6 字段结构化抽取 + 领域分类 | **换 flash + prompt 裁剪**（步骤 2） |
| 2 | 政策回答 `nodes.py:185`（policy_answer_llm） | structured ainvoke | 7/84（S05） | 政策回答 + 引用合规 | **保留 max**，限长 + 分段推送（步骤 3/4） |
| 3 | 追问订单号 `refund.py:146` | astream | 7/84（S04） | 固定话术 | **确定性模板消灭**（步骤 1） |
| 4 | 追问原因 `refund.py:175` | astream | eval 中 0 次（休眠路径） | 固定话术 | **确定性模板消灭**（步骤 1） |
| 5 | 确认话术 `refund.py:225` | astream | eval 中 0 次（休眠路径） | 状态插值话术 | **确定性模板消灭**（步骤 1） |
| 6 | intent_classifier `nodes.py:312` | structured ainvoke | 生产中几乎不可达（active_domain 已由 extractor 预填） | 4 类分类 | **换 flash + 删 few-shot**（步骤 2/3） |
| 7 | 通用 generate `nodes.py:258` | astream | 仅 OTHER 闲聊兜底 | 自由生成 | **换 flash + prompt 精简**（步骤 2/3） |
| 8 | summarizer `state_manager.py:437` | structured ainvoke | ≥6 轮才触发（eval 不触发） | 对话压缩 | 换 flash（低风险，靠单测验证） |

补充事实：

- LLM 实例化仅两处：`nodes.py:32-37` 全局 `llm`；`state_manager.py:107-114` `ConversationStateManager` 内部 `base_llm`（extractor + summarizer）。模型名由 `.env` 的 `LLM_MODEL=qwen3.7-max` 统一提供。
- ORDER 子图（`app/graph/workflows/order.py`）**零 LLM**，订单回答已是确定性合成，不涉及。
- S06 越权拒绝已是确定性文案（`refund.py:139-143`），不涉及 LLM。
- `nodes.py:40-55` 的 `PROMPT_TEMPLATE` 是旧 RAG 模板，`generate()` 已不引用，属死代码。

---

## 3. 四个方向的评估结论

### 方向 1：模型分层 —— ✅ 采纳，收益最大

**不做全局替换**。政策回答是唯一安全关键路径（引用经 `PolicyAnswerGuard` 校验，flash 指令遵循弱会触发校验失败 → 重试延迟翻倍），保留 max；其余轻任务全部降 flash。

| 调用 | 模型 | 理由 |
|---|---|---|
| #1 extractor | qwen3.7-flash | schema 简单；`state_manager.py:323-333` 已有关键词确定性兜底，flash 失败不影响可用性；每轮调用，最大痛点 |
| #3/#4/#5 退款话术 | （步骤 1 直接消灭，不换模型） | — |
| #6 intent_classifier | qwen3.7-flash | 4 分类 + Literal 约束 |
| #7 闲聊 generate | qwen3.7-flash | 兜底路径 |
| #8 summarizer | qwen3.7-flash | 低频；e2e 不触发，靠单测验证 |
| #2 policy_answer | **保留 qwen3.7-max** | 引用合规校验 + 重试机制对指令遵循要求高；仅 7/84 调用量。P2 完成后可选做 max→plus A/B |

### 方向 2：Prompt 优化 —— ✅ 采纳，最大亮点是「消灭调用」

- **退款三话术节点确定性模板化**（步骤 1）：三处输出本质是「固定话术 + 状态插值」，且 fallback 文案本就存在。S04 eval 断言只检查回答含「订单号/单号」关键词（`scripts/eval_agent_e2e.py` 的 `_assert_ask_for_order_sn`），模板天然满足。
- **extractor prompt 两处问题**：① `state_manager.py:317` 把整个 memory JSON 注入 prompt，第二轮起 `last_tool_result` 含 14 项 tool_outcome 元数据，prompt 从 238 涨到 730 tokens——抽取只需 `active_domain / active_order_sn / conversation_summary / collected_slots / pending_slots`，改为白名单投影；② 指令文本 ~200 字压缩到 ~100 字（枚举值保留）；配 `max_tokens=300`。
- **INTENT_PROMPT（`nodes.py:270-284`）**：8 个内联 few-shot 全删（structured output + Literal 下不需要），「只返回标签」规则删除。
- **POLICY_GENERATE_SYSTEM_PROMPT（`nodes.py:57-66`）**：加「answer 不超过 120 字」，配 `max_tokens=700`（必须跑 S05 验证 JSON 不被截断），6 条规则合并为 4 条。
- **GENERATE_SYSTEM_PROMPT（`nodes.py:107-115`）**：删风格词、合并规则。
- 死代码 `PROMPT_TEMPLATE`（`nodes.py:40-55`）删除。
- 全项目无其他 few-shot，「少用示例」策略仅 INTENT_PROMPT 一处适用。

### 方向 3：政策问答流式 —— ⚠️ 只做「校验后分段推送」

`chat.py:142-146` 丢弃 `POLICY_GUARD_TAG` 的流式 token 是**有意安全设计**：未通过引用校验的内容（JSON 片段、违规承诺）不能泄露，校验失败还要重试。真流式（边生成边推）与该校验冲突，不做。

采用**方案 A**：`chat.py:165-167` 目前把 fallback_answer 整段一次性 yield，改为按 ~20 字切片逐片 yield（`asyncio.sleep` 小间隔）。总延迟不变，用户从「等 ~15 秒出整段」变为「生成完成后逐字出现」，零安全风险、成本极低。

> 备选方案 B（flash 先抽引用 → astream 自由生成 → 事后正则校验）属安全权衡，本方案不采纳；如步骤 4 后感知仍差再单独立项。

### 方向 4：Prompt 缓存 —— ✅ 可行但收益最小，列为 P3 可选

已核实 DashScope（OpenAI 兼容模式）缓存机制：

- **隐式缓存**：自动开启、无需配置，公共前缀自动缓存（不保证命中），命中按 20% 计费——零成本自动享受。
- **显式缓存**：messages content 块加 `"cache_control": {"type": "ephemeral"}`，**最小 1024 tokens**，TTL 5 分钟；tools 定义算 system prompt 一部分参与缓存；qwen3.7-flash / qwen3.7-max 均在支持列表。

判断：**缓存只省 prefill，不省生成；本项目瓶颈在 completion**。extractor 固定前缀仅 ~150 tokens 且 memory 每轮变化，不可行；唯一落点是政策路径——`POLICY_GENERATE_SYSTEM_PROMPT` + PolicyAnswer tool schema + `load_policy_rules()` 全文三者跨轮稳定。步骤 5（P3，可选）：先测量三者 token 数，≥1024 才在 `nodes.py:165-172` 构造消息时打 cache_control 标记；不足则只依赖隐式缓存。

---

## 4. 实施步骤（按序执行，每步独立可验证）

### 步骤 1（P0）：退款三话术节点确定性模板化

文件：`app/graph/workflows/refund.py`

- `node_identify_order`（123-161 行）：删除 `llm.astream` 分支，缺单号时直接返回
  `「请提供要退货的订单号（例如 SN20240001）。」`（现有 fallback 文案）。
- `node_collect_reason`（164-190 行）：删除 LLM 分支，直接返回
  `「请说明退货原因（如：质量问题、尺码不合适、不想要了）。」`。
- `node_await_confirmation`（213-244 行）：改为 f-string 模板，插值 `order_sn / reason / eligibility`（eligibility_message 本就是工具确定文案），结尾固定提示点击「确认提交」；保留现有 fallback 文案风格。
- 三节点删除 `from app.graph.nodes import llm` 导入。
- 不改 FSM 路由、不改工具调用、不改 state 字段。

预期：消灭 #3/#4/#5 调用点；refund_agent p95 从 5965ms 降到 <50ms；S04 仍 2/2 通过。

### 步骤 2（P0）：模型分层 + extractor prompt 裁剪

- `app/core/config.py:45` 附近新增 `LLM_MODEL_FAST: str = "qwen3.7-flash"`；`.env` 增加同名配置项（可回退为主模型）。
- `app/graph/nodes.py:32-37` 旁新增 `fast_llm = ChatOpenAI(model=settings.LLM_MODEL_FAST, temperature=0)`；`intent_classifier`（293 行）改为基于 `fast_llm.with_structured_output(...)`；导出 `fast_llm`。
- `app/graph/nodes.py:258` 通用 generate 的 `llm.astream` 改用 `fast_llm.astream`。
- `app/conversation/state_manager.py:107-114`：extractor 的 base_llm 改用 `LLM_MODEL_FAST`；summarizer 保留独立实例（同样切 flash，注释说明 e2e 不覆盖、靠单测）。
- extractor prompt（308-319 行）：
  - 新增 memory 白名单投影：只传 `active_domain / active_order_sn / conversation_summary / pending_slots` 与 `collected_slots` 的 `order_sn / refund_reason / user_confirmed / refund_submitted_id / user_constraints`；**不传** `last_tool_result / workflow_stage / next_action / active_order_id`。
  - 指令压缩至 ~100 字，枚举值（QUALITY_ISSUE 等 5 类）完整保留。
  - fast_llm 实例配 `max_tokens=300`。

预期：extractor 单次 8-11s → 2-4s；订单查询端到端 8.5s → ~3s；退款每轮省 5-7s。重点观察 S03/S07/S08 槽位抽取正确率。

### 步骤 3（P1）：Prompt 清理与政策回答限长

- `nodes.py:270-284` INTENT_PROMPT：删除 8 个示例与冗余规则，保留 4 类定义。
- `nodes.py:40-55` PROMPT_TEMPLATE：删除死代码。
- `nodes.py:107-115` GENERATE_SYSTEM_PROMPT：删风格词，4 条规则合并精简。
- `nodes.py:57-66` POLICY_GENERATE_SYSTEM_PROMPT：加「answer 不超过 120 字」，规则合并为 4 条；policy_answer_llm 配 `max_tokens=700`。
  - **验证要点**：S05 必须 2/2 通过且 `policy_answer_audit.retry_count` 全为 0（无截断重试）；若出现截断，回退 max_tokens 或放宽字数。

预期：政策生成 completion 1023 → 400-600，省 2-4s；边角路径 prompt 体积下降。

### 步骤 4（P2）：政策回答校验后分段推送

文件：`app/api/v1/chat.py:165-167`

- fallback_answer 推送改为切片（~20 字/片，片间 `await asyncio.sleep(0.02)` 左右），保持 SSE 事件格式不变。
- 不动 `on_chat_model_stream` 的 tag 过滤逻辑（POLICY_GUARD_TAG / INTERNAL_LLM_TAG 继续丢弃）。
- order_workflow 短回答走同一切片路径无副作用（文本短，自然结束）。

预期：总延迟不变，TTFT（用户看到首字）感知改善；8 场景行为与断言不变。

### 步骤 5（P3，可选，终验后再做）：政策规则块显式缓存

- 先测量 `POLICY_GENERATE_SYSTEM_PROMPT` + PolicyAnswer tool schema + policy_rules 全文的 token 数。
- ≥1024：在 `nodes.py:165-172` 的消息构造中，对稳定规则块使用 content blocks 形式加 `cache_control: {"type": "ephemeral"}`（LangChain 消息 content 支持 list 结构）；消息顺序保持 system 居首、块顺序固定。
- <1024：不做显式缓存，仅依赖隐式缓存，记录结论。

---

## 5. 验证协议（**冻结，不得擅自变更**）

> 成本控制要求：每步优化只做小样本快速验证，全部完成后才做全量终验。

1. **每完成一个步骤（步骤 1-4）**：跑且仅跑
   `uv run python scripts/eval_agent_e2e.py --runs 2`
   （8 场景 × 2 轮 = 16 runs），记录：
   - 总通过率与每场景 passes（必须保持全绿；出现失败先定位修复，不进入下一步）；
   - `trace.llm_only_latency` 的 p50/p95、llm_calls 次数；
   - 与基线对比的节点延迟变化（重点：extractor 所在轮次、refund_agent、generate）；
   - S05 的 `policy_answer_audit.retry_count`；
   - 产物 JSON 文件名登记到 §7 记录表。
2. **步骤 4（P2）完成后**：跑且仅跑一次全量终验
   `uv run python scripts/eval_agent_e2e.py --runs 7`
   （8 场景 × 7 轮 = 56 runs，与基线同口径），记录最终数字并沉淀 README。
3. **步骤 5（P3）** 为终验后可选项，实施时单独 `--runs 2` 验证缓存命中（响应中缓存指标），不改变终验结论。
4. **执行纪律**：
   - 不得为图快把 `--runs 2` 改成 1，也不得提前跑 7 轮；
   - 2 轮中某场景失败时，允许对**该场景**补跑 1 轮定位（沿用 `agent-evaluation.md` 小样本规则：`--scenarios Sxx --runs 2`），但不得扩大为全量；
   - 任何对轮次/场景范围的调整，必须先征得用户同意；
   - 每步只改该步骤清单内的文件与逻辑，不顺手做清单外改动。

---

## 6. 风险与回退

| 风险 | 应对 |
|---|---|
| flash 结构化抽取/分类质量下降（S03/S07/S08 槽位错误） | extractor 有关键词确定性兜底（`state_manager.py:323-333`）；`.env` 把 `LLM_MODEL_FAST` 改回主模型即可整体回退 |
| flash 的 function calling 兼容性 | 步骤 2 的 `--runs 2` 首验；若 structured output 异常率高，extractor/classifier 单独回退 max |
| 政策 max_tokens=700 截断 JSON | 步骤 3 验证 retry_count；出现截断立即回退该参数 |
| 模板话术影响 S04 断言 | 断言只查「订单号/单号」关键词，模板已包含；步骤 1 首验确认 |
| 分段推送改变前端渲染 | SSE 事件格式不变，仅拆片；前端按 token 拼接逻辑已存在 |

---

## 7. 执行记录表

| 步骤 | 内容 | 验证命令 | 产物 JSON | 通过率 | llm_only p50/p95 | 备注 |
|---|---|---|---|---|---|---|
| 基线 | — | — | agent_e2e_20260907_084355.json | 56/56 | 9617 / 15677 ms | qwen3.7-max；llm_calls=84（stream 7） |
| 1 | 退款话术模板化 | `--runs 2` | agent_e2e_20260907_105130.json | 16/16 | 9078 / 15854 ms | identify_order 4726/6404ms→1.1/1.4ms；refund_agent p95 5965→225ms；exact_stream 7→0；单测 48 全过；S04 模板断言通过 |
| 2 | 模型分层 + extractor 裁剪 | `--runs 2` | （待填） | | | |
| 3 | Prompt 清理 + 政策限长 | `--runs 2` | （待填） | | | |
| 4 | 政策回答分段推送 | `--runs 2` | （待填） | | | |
| 终验 | P2 全量 | `--runs 7` | （待填） | | | 最终数字沉淀 README |
| 5（可选） | 政策显式缓存 | `--runs 2` | （待填） | | | token 测量结论： |
