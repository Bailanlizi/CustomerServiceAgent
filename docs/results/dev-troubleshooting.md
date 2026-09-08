# 开发过程问题排查记录（Troubleshooting Log）

> 本文档记录本项目（E-commerce Smart Agent）在开发、评估、优化过程中**真实遇到的问题、尝试过的方案、最终解决方法与可量化的效果**。
> 定位：内部复盘 / 面试素材，与 `RESUME_NOTES.md` / `INTERVIEW_QA.md` 配套，但**不重复**它们的"项目亮点"叙述，只聚焦"踩坑与解决"。
> 所有条目均可回溯到代码 / `docs/` 设计文档 / `eval/runs/` 评估产物的具体位置。
> 原则：**诚实记录**——已解决、部分解决（含遗留边界）、刻意不解决（标注理由）都写清楚，不把"能跑通"包装成"无问题"。

---

## 0. 速览（按模块归类）

| 模块 | 问题数 | 代表性问题 |
|---|---|---|
| 评估脚本口径 | 5 | 工具指标只取末轮 / astream token 漏记 / 节点延迟双计 / S06 假拦截 / S08 测错层 |
| LLM 延迟优化 | 6 | flash 默认 thinking 截断 JSON / 退款话术过 LLM / extractor 全量记忆进 prompt / 安全路径真流式 |
| RAG 子系统 | 7 | 条款被切碎 / 文档级规则稀释 / 层级冲突位次差 / FAQ 重排置顶 / RAGAS judge 超时 / faithfulness 偏差 / 多跳题天花板 |
| 会话与持久化 | 1 | 刷新/重登后历史气泡丢失 |
| 工程与稳定性 | 4 | 进程内幂等缓存污染 / 跨场景 SN 串号 / 11 次 run 混退化数据 / 评估口径早于断言 |
| 架构与工具治理 | 3 | 工具名枚举全错 / read 工具不写审计 / 越权靠 LLM 拒答无硬拦截 |

---

## 1. 评估脚本口径（Agent 端到端评估）

### 1.1 工具指标只取最后一次 run，却声称 N=5

- **问题**：`scripts/eval_agent_e2e.py:646` 的聚合逻辑 `tool_results[s.id] = all_runs[s.id][-1]` 只保留每个场景**最后一次** run 的工具结果；而 `normal_success_rate` 是跨 N 轮聚合的。报告写"N=5"，但 precision/recall/order_accuracy 实际只基于 1 次 run，**口径自相矛盾**（`agent-evaluation-revision.md` P0-1）。这比延迟 bug 更该先修——它直接动摇"工具选择准确率 1.0"的可信度。
- **尝试**：先怀疑是 LLM 非确定性导致波动，后发现温度已固定为 0（`nodes.py:36` / `state_manager.py:111`），排除采样因素。
- **解决方法**：Phase A1——遍历每场景全部有效 run 逐次算 precision/recall/order_accuracy，输出 `mean/min/max/n`；过滤 `llm_calls < 3` 或 turns 不完整的退化 run（保留但不计入指标）。
- **效果**：修复后 `tool_metrics` 含 `n`（precision n=35 / recall n=21 / order n=42，均 1.0），`degenerate_runs` 全 0；口径从"1 次快照"变为"跨 N 分布"，数字可解释（`eval/runs/agent_e2e_20260907_084355.json`）。

### 1.2 astream token 系统性漏记，报告 token 是下界

- **问题**：`eval_agent_e2e.py:109-112` 的 `install_token_callback` 对 `ChatOpenAI.astream` 只记 `{"kind":"astream","prompt_tokens":None,"completion_tokens":None}`。流式调用 token 被丢弃，且**丢的恰是最贵的大生成调用**（退款话术、政策生成）。报告 `total_prompt=23787 / completion=27368` 是下界（`agent-evaluation-revision.md` P0-4）。
- **尝试**：核对是否 LangChain 流式不返回 usage——确认 OpenAI 协议末 chunk 带 `usage_metadata`，需显式累加。
- **解决方法**：A2——流式末 chunk 读 `usage_metadata`，或在 `ChatOpenAI` 构造时传 `stream_options={"include_usage": true}`；输出 `token_source ∈ {exact_stream, exact_invoke, unknown}` 便于校验。
- **效果**：补全后 token 总量 51155 → **74347（+45%）**，证实旧版系统性低估；`token_source_counts.unknown == 0`（`084355` 产物）。

### 1.3 节点耗时双计 + 归因错位，refund 延迟被严重低估

- **问题**：`041503.json` 中 `generate p50=14987.8` 与 `RunnableSequence p50=14986.8`（差 1ms，同一次调用）被计两次；`S06 identify_order=0.4ms` 却产 522 token（物理不可能）。`node_latency` 把 astream 外层+内层双报、把 ainvoke 的 wall-clock 漏归到节点（`agent-evaluation-revision.md` P0-5）。`refund_agent p50=24ms` 不可信。
- **尝试**：最初直接读 `astream_events` 的节点包裹耗时，发现包装节点（RunnableSequence/Lambda）与业务节点同名混报。
- **解决方法**：A3——聚合时过滤 LangChain 内部包装节点，只保留业务语义节点（dispatch_router / order_workflow / refund_agent / identify_order / …）；在 `_ainvoke` patch 内同步记 `perf_counter` 起止，显式归 wall-clock；新增独立的 `llm_only_latency` 指标。
- **效果**：双计消除，`llm_only_latency` p50=9617ms / p95=15677ms（n=84），与 `generate` 节点 p50=15.7s 对齐（`084355`）。

### 1.4 S06 越权是"LLM 假拒绝"，未走权限层

- **问题**：`refund.py:138-143` 的 `node_identify_order` 情形2，当订单号存在但 `active_order_id` 缺失时**直接吐固定文案**，零工具调用（trace `tool_names=[] codes=[]`）。`security_intercept_rate=1.0` 测的是"SQL 反查 + LLM 确定性拒答"，**未经 `ToolCapabilityRegistry` 的 `NOT_AUTHORIZED` code 路径**（`agent-evaluation-revision.md` P0-2）。
- **尝试**：想把 S06 改成"强制 agent 调 `query_order_tool` 断言 `NOT_AUTHORIZED`"——但评估脚本无法凭空让节点产 code，须改业务代码。
- **解决方法**：B2——评估侧保留断言，但把报告口径**改名为"间接安全信号"（indirect）**，`summary` 显式标注 `assertion_type="indirect"` + `assertion_note`；业务侧 D1（单列，未实施）才是真正下沉权限层。
- **效果**：报告不再出现"安全拦截率 1.0"这种暗示经权限层拦截的措辞；诚实标注为 indirect（7/7 仍全过，但语义清晰）。

### 1.5 S08 幂等测的是 eligibility 层，不是 submit 层

- **问题**：5 次 run 全是 `codes=['ELIGIBILITY_REJECTED']`，从没命中 `ALREADY_EXISTS/REFUND_SUBMITTED`。`submit_refund_application` 本身是否幂等**完全没测到**；且 `allowed_chains=[]` 把合理的 `check_refund_eligibility` 判成"多余"，导致 precision 0.857 / order 0.833 是**假负例**（`agent-evaluation-revision.md` P0-3 / P1-7）。
- **尝试**：方案①（纯评估脚本改 `allowed_chains`）vs 方案②（真测 submit 层幂等，依赖业务改造）。
- **解决方法**：B1——采纳方案①：`allowed_chains=[[TOOL_CHECK_ELIGIBILITY]]`，承认 eligibility 是合理前置；断言保留 `refund_count==1` + 答案含"已有/已存在"。
- **效果**：precision / order_accuracy 假负例消除，剔除 S08 后均 = 1.0；报告可解释 S08 测的是"eligibility 层拦截"还是"submit 层幂等"。

---

## 2. LLM 延迟优化

### 2.1 换 flash 模型后首验 16 runs 全挂：默认 thinking 截断 JSON

- **问题**：步骤 2 把 extractor/intent_classifier 切到 `qwen3.7-flash` 后，首验产物 `125919` 中 **S03/S07/S08 全挂**。根因：`qwen3.7-flash` **默认开启 thinking**，结构化调用输出 550-900+ tokens 推理文本，把 `max_tokens` 预算烧光 → JSON 截断异常 → 静默走关键词兜底（"我要退 SN…" 不含"退款/退货" → 误判 OTHER → FSM 停在 collect_reason）（`llm-latency-optimization-final.md` §四 步骤2）。
- **尝试**：先调大 `max_tokens`（300→900）试图兜住推理文本，无效——推理文本仍可超预算。
- **解决方法**：快模型统一加 `extra_body={"enable_thinking": False}`。实测 output 550→111 tokens，单次延迟 3.4s→0.8s。
- **效果**：修复后 16/16 全通过；`llm_only` p50 9078→894ms，completion 总量 11039→3211，prompt 总量 9520→6564。
- **教训（已写进文档）**：换快模型必须先单独直连测 structured output 的真实 completion 开销，再定 max_tokens。

### 2.2 退款三话术节点用 LLM 生成，慢且不稳定

- **问题**：`node_identify_order` / `node_collect_reason` / `node_await_confirmation` 三处用 `llm.astream` 生成追问/确认话术，单点 `identify_order` 4726/6404ms，`refund_agent` p95 5965ms。且 eval 56 个 run 里这三类话术节点**一次都没触发**（本质固定文案 + 状态插值）。
- **尝试**：考虑保留 LLM 以"话术更自然"——但 extractor 已预填订单号/原因，FSM 每轮只走一个阶段，工具结果自带确定性文案。
- **解决方法**：步骤 1（P0）——三节点全改固定模板字符串 f-string 插值，删 `from app.graph.nodes import llm` 导入。
- **效果**：`identify_order` p50 4726→1.1ms，`refund_agent` p95 5965→225ms（终验 50ms），`exact_stream` 7→0；话术一致性反而更好。

### 2.3 extractor 把整段 memory JSON 塞进 prompt，第二轮起暴涨

- **问题**：`state_manager.py:317` 把整个 memory JSON 注入 extractor prompt，第二轮起 `last_tool_result` 含 14 项 tool_outcome 元数据，prompt 从 238 涨到 730 tokens——抽取只需 5 个字段（`app/graph/nodes.py` 实例）。
- **尝试**：只压缩指令文本（~200→100 字），token 下降有限。
- **解决方法**：步骤 2——memory 白名单投影：只传 `active_domain / active_order_sn / conversation_summary / pending_slots` 与 `collected_slots` 子集；不传 `last_tool_result / workflow_stage / next_action / active_order_id`。配 `max_tokens=300`（修正后实际用 900，因关 thinking 后余量充足）。
- **效果**：prompt 稳定在 ~150 tokens，extractor 单次 8-11s→2-4s。

### 2.4 政策回答路径 max_tokens 硬限未加（刻意）

- **问题**：方案原本计划给 `policy_answer_llm` 加 `max_tokens=700` 硬限防 JSON 超长。但结构化 JSON 输出截断会触发 `PolicyAnswerGuard` 校验失败 → 重试延迟翻倍（方案预警的风险）。
- **尝试**：评估两种约束——硬限（max_tokens）vs 软约束（prompt 写"≤120 字"）。
- **解决方法**：步骤 3——**不加硬限**，靠 prompt 软约束 + 单独构造 `_policy_llm` 实例关 thinking（max 模型 thinking 开/关单条 9.8s→2.7s）。
- **效果**：S05 completion 1023→103，`retry_count=0`（无截断重试）；这是 p95 下降的核心。`generate` 节点 p50 16924→2614ms。

### 2.5 政策回答"真流式"与安全校验冲突

- **问题**：`chat.py:145-146` 丢弃 `POLICY_GUARD_TAG` 的流式 token 是有意安全设计——未通过引用校验的 JSON 片段/违规承诺不能泄露。真流式（边生成边推）会让未校验内容先吐给前端，违反安全设计。
- **尝试**：备选方案 B（flash 先抽引用 → astream 自由生成 → 事后正则校验）属安全权衡。
- **解决方法**：步骤 4——只做"校验后分段推送"：`fallback_answer` 按 ~20 字切片、片间 `asyncio.sleep(0.02)` 逐片 yield，SSE 事件格式不变。
- **效果**：总延迟不变（预期），TTFT 感知改善；零安全风险，SSE 单测 67 全过。

### 2.6 全局单一 max 模型，延迟全由 LLM 支配

- **问题**：基线 `llm_only` p50=9617ms / p95=15677ms，84 次 LLM 调用中 92%（77 次）走 `ainvoke`，首 token = 总延迟；政策生成单次 completion 达 1023 tokens（`llm-latency-optimization.md` §1）。
- **尝试**：考虑全局换 flash——但政策回答是安全关键路径（引用经 Guard 校验，flash 指令遵循弱会触发校验失败 → 重试延迟翻倍）。
- **解决方法**：方向 1（模型分层）——政策回答保留 max，其余高频轻任务（extractor/intent/classifier/闲聊/summarizer）降 flash；删 `INTENT_PROMPT` 8 个 few-shot。
- **效果**：终验 `llm_only` p50 910ms（-91%）/ p95 2405ms（-85%），completion 总量 40533→5493（-86%）。

---

## 3. RAG 子系统

### 3.1 用通用切分器会切碎条款、丢失编号边界

- **问题**：政策文件若用 `RecursiveCharacterTextSplitter` 按字符切，会把 `## RETURN_001: 7天无理由退货` 这类条款切散，编号边界丢失——而**编号是评估命中锚点 + 引用校验锚点**（`rag-knowledge-and-evaluation.md` §2.2 / §7）。
- **尝试**：先按字符切 + 后置正则重建编号，复杂且易错。
- **解决方法**：`policy_chunks.py:load_policy_documents` 按正则 `^##\s+([A-Z]+_\d{3}):\s*(.+?)$` **以条款标题为边界**，一条一个 chunk，元数据严格保留 `clause_ids/clause_title/source/...`。
- **效果**：条款级召回可精确评估；编号成为确定性校验最小单位。

### 3.2 文档级规则复制到每条，稀释条款向量

- **问题**：每份政策文件头部有"优先级/适用关系"的文档级规则。若把规则文本复制到该文件每个条款 chunk，会稀释条款本体的向量语义，检索时文档级规则还可能挤占事实条款位次。
- **尝试**：将规则作为普通 chunk 一并入库（最早实现）。
- **解决方法**：文档级规则**单独入库**（source_type=`policy_rule`），不复制到每条；检索时 `policy_rule` **不进检索位次**（仅作生成期背景喂给模型）（`rag-knowledge-and-evaluation.md` §2.2/§2.4）。
- **效果**：P0 阶段2——`层级冲突` primary_hit 0.833→1.000，recall 0.833→0.917（retrieval_p0_final）。

### 3.3 层级冲突题位次质量差（MRR 0.51）

- **问题**：基线 `层级冲突` 题型 primary_hit 0.833、mrr 0.514——单一维度命中到位，但跨文档优先级冲突（如 VIP 例外 vs 通用规则）题位次差（`rag-knowledge-and-evaluation.md` §5 P0-1）。
- **尝试**：直接调 top_k，收效有限。
- **解决方法**：P0 阶段2 引入 `source_type` 区分 + `policy_rule` 不进检索 + 相似度阈值 `<0.5` + 候选池扩到 `max(top_k*4,20)`。
- **效果**：`层级冲突` primary_hit→1.0；整体 mrr 0.783→0.792。

### 3.4 FAQ 来源感知重排"粗暴置顶"会牺牲多跳题

- **问题**：FAQ 转述常引用权威条款（如 FAQ_011 引用 CAT_001）。若把 FAQ 引用的权威条款粗暴置顶，会挤掉多跳题（如"内衣试穿+黑卡+VIP"）的必要次要证据，降低召回。
- **尝试**：先试"FAQ 命中即置顶"——多跳题 recall 下降。
- **解决方法**：P1 阶段1——`_rerank_by_authority`：把被 FAQ 引用的权威条款**只升到对应 FAQ 的前一位**，其余保留原始语义排序（`rag-knowledge-and-evaluation.md` §2.4）。
- **效果**：mrr 0.7915→**0.8685**（+0.077）；easy mrr 0.671→0.907；**hard mrr 微降 0.756→0.653**（有意识取舍，docstring 已写明代价）。

### 3.5 RAGAS Faithfulness judge 默认 thinking 触发超时

- **问题**：`deepseek-v4-flash` 作 RAGAS judge，默认开 CoT，单次 <1s 涨到 10-30s，叠加 RAGAS 每题 4 次调用频繁超时（120s 上限）（`rag-knowledge-and-evaluation.md` §4.3）。
- **尝试**：增大超时——不可持续（总成本与延迟失控）。
- **解决方法**：judge 强校验 `extra_body={"thinking":{"type":"disabled"}}`；强制 judge ≠ 业务 LLM（防自评高估）；prompt 改用中文（`chinese_ragas_prompts.py`）。
- **效果**：faithfulness 0.9091（oracle）/ 0.9347（baseline），运行稳定。

### 3.6 RAGAS 默认英文 prompt + factual_correctness 系统性误判

- **问题**：RAGAS 默认英文 prompt 对中文场景 faithfulness 跑偏；`factual_correctness` 模式只罚"答案里有而 ground_truth 没有的 claim"，客服答案天然比一句话 ground_truth 长（条款引用+客套话+追问），导致**全对答案被系统性判低分**（曾有 4 个 0 分题答案与 ground_truth 完全一致）。
- **尝试**：调高 factual_correctness 权重——仍误判。
- **解决方法**：不评测 factual_correctness，改由确定性的 `answer_clause_metrics` 衡量生成侧事实正确性；faithfulness 仅衡量"是否忠于检索上下文"。
- **效果**：消除系统性误判；faithfulness 数字可信（但仍保留 6% 答案被判"不忠"的诚实记录，多为补充话术）。

### 3.7 Embedding 端点未通，整轮 ETL 失败

- **问题**：早期 P0 跑 `etl_policy_p0.log`，6 个文件全部 `APIConnectionError`（Embedding 端点未通），知识库零入库（`rag-knowledge-and-evaluation.md` §2.3）。
- **尝试**：重跑——仍失败。
- **解决方法**：切换 Embedding 端点后重跑；`etl_policy.py` 改为 50 条/批 + 3 次指数退避重试 + 旧数据幂等清理（`DELETE WHERE source=当前文件`）。
- **效果**：成功入库；失败 run 日志保留为可复现证据。

### 3.8 多跳/层级冲突题真实天花板（Q001）

- **问题**：hard 题里 **Q001「黑卡会员买内衣试穿过了还能退吗」**（expected `VIP_001/VIP_006/CAT_001`）mrr=0.25、recall<1。单 query embedding 在"试穿+黑卡+内衣"三语义信号上无法同时对齐三条款——这是**单 query embedding + 候选池 20 + Top-K 重排单跳设计的真实天花板**（`rag-knowledge-and-evaluation.md` §6.3/§6.4）。
- **尝试**：调 rerank 策略、扩 top_k——均以牺牲中位数为代价，对 hard 题帮助有限。
- **解决方法**：**刻意不刷分**，列入 Phase D 候选（多向量召回 / HyDE / 跨文档图谱检索）。
- **效果**：诚实标注短板，避免"调 prompt 把数字做漂亮"——这是金融科技/严肃系统应有的严谨性。

---

## 4. 会话与持久化

### 4.1 刷新/重登后历史气泡丢失

- **问题**：客服窗口气泡仅存在于 Gradio `chatbot` 内存（`customer_ui.py:390` `handle_logout` 返回 `[]`、`handle_login` 不回填）；刷新或退出再登录，历史消息被清空。用户期望"重登后历史都在且能续聊"（`chat-history-recovery.md` §1）。
- **尝试**：先确认 LLM 记忆连续性——发现 `ConversationSession.checkpoint_thread_id` + `working_memory_json` 已能复用（同浏览器已满足）；真正缺口是"气泡可见性"与"登录后拉历史接口"。
- **解决方法**：新增 `conversation_messages` 表 + `GET /api/v1/chat/session` 接口（按 `user_id` 取 `last_active_at DESC LIMIT 1`，无则建）；`handle_login` 后回填 history；服务端 `event_generator` 自行拼接最终回复且只落库一次；`resolve_session` 当 `conversation_id=None` 时统一走 `user_id` 取默认行；首次创建用 `IntegrityError` 重查处理竞态（§4-§8）。
- **效果**：验收 1/2（刷新后仍在、重登自动显示）满足；跨设备顺带覆盖；`user_id` 隔离复用现有 `Forbidden` 防护。**范围决策：不做"新建会话"**（一个用户一个 append-only 窗口）。

---

## 5. 工程与稳定性

### 5.1 进程内幂等缓存未清理，导致 N=2 假失败

- **问题**：`submit_refund_application` 走 `tool_registry._idempotency_cache`（key `refund:{user_id}:{order_id}`）。测试 `cleanup_order_refund` 只删 DB 行、不删缓存 → run 2 的 submit 命中缓存返回 `REFUND_SUBMITTED` 却不再落库 → DB 无记录 → S03/S07 status=None、S08 幂等断到 0 条（`2026-09-07.md` 调试踩坑 1）。
- **尝试**：怀疑 LLM 非确定性——温度已固定 0，排除。
- **解决方法**：cleanup 删 DB 前先 `tool_registry._idempotency_cache.clear()`（整体清空，因 key 含 order_id 无法仅凭 order_sn 反推）。
- **效果**：N=2→N=5 连续全 PASS，数字稳定。

### 5.2 跨场景 SN 串号，越权场景误判

- **问题**：S06（越权）与 S07（重登退款）原来共用 `SN20240004`，N=2 run 2 时越权误判（状态污染）。
- **尝试**：统一 seed 数据隔离。
- **解决方法**：S06 改用 `SN20240005`、S07 用 `SN20240004` 隔离。
- **效果**：跨场景不再串号，越权断言稳定。

### 5.3 11 次 run 混退化数据，"全 PASS"是靠筛选

- **问题**：重算 `eval/runs/` 发现 11 次运行里仅 7 次是完整 8 场景真实 LLM 运行，3 次是调试（仅 S01/S03，`llm_calls=0~2`）、2 次早期低调用。且早期 7 次里 S03/S06/S07/S08 曾整轮失败（min=0.0）。干净结论是复跑筛选出来的（`2026-09-07.md` 客观重分析）。
- **尝试**：最初直接报"40/40 全 PASS、无残余波动"——被交叉审查纠正。
- **解决方法**：Phase A1 过滤退化 run + 跨 N 聚合 + 声明小样本置信区间；后续修复幂等缓存+SN 隔离后，`040328` 起连续稳定。
- **效果**：口径从"快照/筛选"变为"跨 N 分布 + 诚实区间"；但 7 次样本仍小，真稳定需 30-100 次（已标注）。

### 5.4 评估口径先于断言：第三方审查暴露 4 个硬伤

- **问题**：初版评估方案工具名全错（`query_order_tool` 等注册名 vs 草稿写的 `query_order`）、`RefundStatus` 无 `SUBMITTED`（FSM 用 `RefundStage`）、`AuditAction` 无 `PERMISSION_DENIED`（越权源是 `ToolCode.NOT_AUTHORIZED`）、主图节点链不符（`2026-09-07.md` 第三方审查核实）。
- **尝试**：逐条只读核实——4 条硬伤全部成立。
- **解决方法**：方案 v2/v3 用真实工具名/枚举/节点链重写；工具链采集改 hook `ToolCapabilityRegistry.invoke`（因 read 工具不写 audit_log，只读表会漏掉 query_order_tool 等）；token 埋点覆盖 2 个 LLM 实例。
- **效果**：方案从"草稿"变为"代码可核实"；净复杂度不增反降（场景 12→8 省的时间 > hook 增加的时间）。

---

## 6. 架构与工具治理

### 6.1 工具越权靠 LLM 拒答，无权限层硬拦截

- **问题**：S06 越权当前路径是 prepare_turn 的 SQL 反查（查不到他人订单）+ identify_order 固定文案直接拒答，**零工具调用**（`tool_names=[] codes=[]`）。系统没有"经 `ToolCapabilityRegistry` 返回 `NOT_AUTHORIZED`"的硬拦截路径（见 1.4）。
- **尝试**：评估侧只能诚实标注 indirect（见 1.4 B2）。
- **解决方法（业务侧，Phase D 未实施）**：D1——改 `node_identify_order` 情形2 或 prepare_turn 反查逻辑，让"查询他人订单"显式调 `query_order_tool` 触发 `NOT_AUTHORIZED` code；S06 断言才能升级为硬断言（`agent-evaluation-revision.md` §Phase D）。
- **效果（当前）**：间接安全信号 7/7，但真实权限层拦截未验证——面试口径必须诚实说明。

### 6.2 工具审计只覆盖 sensitive，read 工具不落 audit_log

- **问题**：`audit_level="read"` 不落库，只有 sensitive 的 `submit_refund_application` 写 `audit_logs`。若只从 `audit_logs` 表采集"实际工具链"，只能拿到 submit 一条，query_order_tool/check_refund_eligibility/query_refund_status 全漏（`2026-09-07.md` 补充发现 2）。
- **尝试**：先想靠审计表反推工具链——不可行。
- **解决方法**：工具链采集改为 hook `GuardedToolExecutor` / `ToolCapabilityRegistry.invoke`（唯一覆盖点 `tool_registry.py:139`），一处覆盖全部 4 工具 + 所有调用路径。
- **效果**：工具调用 oracle 与真实链可追溯；`ToolOutcome` envelope 承载 `code/metadata` 供审计与评估复用。

### 6.3 退款支付是 mock，未接真实网关

- **问题**：退款子流程的支付环节是 mock 实现，尚未接入真实支付网关（README 已标注）。接前应补支付流水、网关幂等键、第三方交易号持久化。
- **尝试/解决**：作为已知边界明确标注，列入后续方向（不在本轮设计）。
- **效果**：当前端到端评估集中在"业务闭环正确性"，资金链路为演示级——面试口径诚实声明"mock 支付"。

---

## 7. 沉淀的方法论（可迁移）

1. **评估口径必须先于断言**：先修采集口径（跨 N 聚合 / token 补全 / 去双计），再修场景断言（S06/S08），最后才加难场景——否则在错误口径上继续制造假结论（`agent-evaluation-revision.md` §5）。
2. **换快模型先直连测 structured output 的真实开销**：thinking 默认开是 qwen3 系列快模型的隐藏坑，必须先 `enable_thinking: false` 再定 `max_tokens`。
3. **确定性输出不过 LLM**：固定话术/状态插值用模板，既快又稳。
4. **安全路径不做真流式**：未校验 token 不能泄露，用"校验后分段推送"折中。
5. **judge 必须与业务 LLM 隔离**：自评会系统性高估，是 RAG 评估的硬约束。
6. **诚实标注能力边界**：S06 indirect / S08 eligibility 层 / Q001 多跳短板 / mock 支付 / 小样本——不把"能跑通"包装成"系统确定可靠"，这是金融科技/严肃系统应有的严谨性。

---

## 8. 仍遗留 / 刻意未解决（面试前须知）

| 项 | 状态 | 说明 |
|---|---|---|
| S06 真 Guard 断言（NOT_AUTHORIZED 硬拦截） | Phase D 未实施 | 当前 indirect 验证 |
| S08 submit 层幂等（重复调是否重复落库） | Phase D 未实施 | 当前测 eligibility 层 |
| 参数准确率（order_id/user_id 来自认证上下文） | 缺失 | P2-10，未校验注入参数来源 |
| 测试集 45 题 | 单域小样本 | 置信区间宽，扩 100-200 + 多评估者交叉标注 |
| 多跳/层级冲突题检索天花板 | 真实上限 | 单 query embedding 设计，需多向量召回/HyDE |
| 真实支付网关 | mock | 演示级资金链路 |
| 文档状态与代码脱节 | 待修 | `agent-evaluation-revision.md` / `llm-latency-optimization.md` / `chat-history-recovery.md` 顶部仍写"待实施"，实际已完成/已实现 |

---

*本文档所有条目均来自开发过程中的真实记录（代码静态分析 + `eval/runs/` 产物 + `docs/` 设计文档 + 工作日志），未编造。*
