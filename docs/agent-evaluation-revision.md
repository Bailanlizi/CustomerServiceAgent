# Agent 评估模块修正方案

> 状态：**待实施**（2026-09-07）
>
> 本文是 [agent-evaluation.md](./agent-evaluation.md) 的修正补充，**不重复**口径定义，只针对最新一轮评估产物 `eval/runs/agent_e2e_20260907_041503.json` 暴露的问题给出收敛后的修复计划。所有结论由三轮交叉审查（重算 + 两个独立审查 agent + 代码静态核对）支撑，依据行号见各条标注。
>
> 真源边界：本文档只动评估脚本与场景定义；**凡需要改业务代码的项，单独标注为业务改造项，不混入评估脚本修复清单**。

---

## 1. 评估现状一句话

**实现质量高，结果口径有硬伤**：事实硬伤（工具名/枚举/节点链）已全部修正，8 场景 5 runs 全 PASS，但**这个 100% 是评估口径造出来的干净，不能直接对外讲成"系统可靠"**。需要按本方案修正采集与断言口径后复跑。

---

## 2. 已核实问题清单（按严重度，附行号证据）

### 严重（直接决定结果是否讲真话）

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| P0-1 | **工具指标只取最后一次 run** | [eval_agent_e2e.py:643-647](file:///d:/PythonProject/CustomerServiceAgent/scripts/eval_agent_e2e.py#L643-L647) `tool_results[s.id] = all_runs[s.id][-1]`；而 `normal_success_rate` 是跨 N 聚合 | 报告声称 N=5，precision/recall/order_accuracy 实际只基于 1 次，**口径自相矛盾** |
| P0-2 | **S06 越权走的是 `identify_order` 节点固定文案，非 Guard 工具层** | [refund.py:138-143](file:///d:/PythonProject/CustomerServiceAgent/app/graph/workflows/refund.py#L138-L143) `order_sn` 存在但 `active_order_id` 缺失 → 直接吐固定文案；trace `tool_names=[] codes=[]` | `security_intercept_rate=1.0` 测的是"SQL 反查 + 确定性拒答"，**未经过 `ToolCapabilityRegistry` 的 `NOT_AUTHORIZED` code 路径** |
| P0-3 | **S08 幂等测的是 eligibility 层，不是 submit 层** | 5 runs 全是 `codes=['ELIGIBILITY_REJECTED']`，从没命中 `ALREADY_EXISTS/REFUND_SUBMITTED` | `submit_refund_application` 本身是否幂等（重复调是否重复落库）**完全没测到** |
| P0-4 | **astream token 系统性漏记** | [eval_agent_e2e.py:109-112](file:///d:/PythonProject/CustomerServiceAgent/scripts/eval_agent_e2e.py#L109-L112) `astream` 记 `None/None`；S04 trace 可见 `('astream', None, None)` | `total_prompt=23787 / completion=27368` 是下界，且漏的恰是最大最贵的退款 LLM 调用 |
| P0-5 | **节点耗时双计 + 归因错位** | `041503.json` 中 `generate p50=14987.8` / `RunnableSequence p50=14986.8`（差 1ms，同一次调用）；S03 `check_eligibility p50=6.8ms` 但该轮 ainvoke 产 533 completion tokens（物理不可能） | `node_latency` 把 astream 外层+内层双报、把 ainvoke 的 wall-clock 漏归到节点，**refund 路径延迟严重低估**（`refund_agent p50=24ms` 不可信） |

### 中等（覆盖面与方法学）

| # | 问题 | 证据 |
|---|---|---|
| P1-6 | 100% 成功率过干净 | 断言大量用答案关键词 fallback（"未查询到"/"已有"），全走 happy path seed 数据，无 adversarial/多跳/歧义 |
| P1-7 | precision 0.857 / order 0.833 是 S08 假负例 | S08 的 5 次 `check_refund_eligibility` 被 `allowed_chains=[]` 判多余；剔除 S08 后两者都 =1.0 |
| P1-8 | 11 次 run 非同质 | 混了 3 次调试（仅 S01/S03，`llm_calls=0~2`）+ 早期低调用 run；仅 7 次是完整 8 场景 |
| P1-9 | 抖动非采样随机 | `nodes.py:36` / `state_manager.py:111` / `evaluate_ragas.py:88` 全 `temperature=0` → 偶发整轮失败来自 harness 状态污染（如 S08 依赖同进程 S03 先建单），不是模型采样 |

### 轻度（优化项，本方案不实施）

| # | 问题 |
|---|---|
| P2-10 | 缺参数准确率（订单号/原因抽取、`user_id/order_id` 是否来自认证上下文） |
| P2-11 | 缺"严格轨迹准确率 vs 业务成功率"双指标 |
| P2-12 | 缺 API/SSE 层 e2e（鉴权、SSE 完整结束、并发、超时重试） |
| P2-13 | RAG 评估无分阶段数据（`run_rag_baseline.py` 只记整体 `duration_seconds`，无 retrieve-vs-generate 拆分） |

---

## 3. 根因分类（一句话归类）

- **结果可信度**（P0-1/P0-2/P0-3/P1-6/P1-7/P1-9）：场景设计 + 断言口径宽松 → 把"能跑通"包装成"100% 可靠"。
- **指标采集口径**（P0-4/P0-5）：评估脚本自身 monkey-patch 对 `astream`/`ainvoke` 处理不对称，导致 token 漏记、延迟归因错位、双计。
- **稳定性**（P1-8/P1-9）：调试 run 与真实 run 混、场景间状态不隔离，数字不可复现。

---

## 4. 修复方案（落实清单）

### Phase A — 让数字本身可信（纯评估脚本，零业务风险）

**A1（P0-1）｜工具指标跨 N 聚合 + 剔退化 run**
- 文件：`scripts/eval_agent_e2e.py` 聚合段（当前 `all_runs[s.id][-1]`）
- 改法：
  1. 遍历每场景全部有效 run 逐次算 precision/recall/order_accuracy，输出 `mean / min / max`
  2. 过滤 `llm_calls < 3` 或 `turns` 不完整的退化 run（不计入指标，但仍保留在 `runs` 供复查）
  3. `summary.tool_metrics` schema 扩展为 `{precision: {mean,min,max}, recall: {...}, order: {...}}`
- 验收：N=5 时三指标都有 5 个采样点；断言聚合函数对 S01-S05/S07 至少输出 5 个值

**A2（P0-4）｜astream token 补全**
- 文件：`scripts/eval_agent_e2e.py` `install_token_callback`（L109-112）
- 改法：
  1. 流式末 chunk 读 `usage_metadata`（LangChain `AIMessage.usage_metadata` 在最后一个 chunk 合并）
  2. 或在 `ChatOpenAI` 构造时传 `stream_options={"include_usage": true}`（OpenAI 协议支持流式返回 usage）
  3. 输出 `llm_calls[].token_source ∈ {exact_stream, exact_invoke, unknown}`，便于事后校验
- 验收：S03/S04/S07 的 astream 调用 token 不再为 None；`total_prompt_tokens` 增长 ≥ 30%

**A3（P0-5）｜节点耗时去双计 + 修归因**
- 文件：`scripts/eval_agent_e2e.py` 的 trace 采集与聚合
- 改法：
  1. 聚合时过滤 LangChain 内部包装节点（`RunnableSequence` / `RunnableLambda` / `RunnableParallel`），只保留业务语义节点（`dispatch_router` / `order_workflow` / `refund_agent` / `identify_order` / `check_eligibility` / `submit` / `retrieve` / `generate` / `compose_answer` / `entry` / `route_from_entry` / `query_order`）
  2. 归因修复：在 `_ainvoke` patch 内同步记 `perf_counter` 起止，把 ainvoke 的 wall-clock 显式归到"当前 LLM 调用上下文"——不能依赖 astream_events 自动归因
  3. 输出 `llm_only_latency` 独立指标，与 `node_latency` 分开报
- 验收：`RunnableSequence` 不再出现在 `node_latency`；`refund_agent` p50 不再是 24ms 量级

### Phase B — 让场景断言可信（评估脚本为主，S06 业务改造单列）

**B1（P0-3）｜S08 重设**
- 文件：`scripts/eval_agent_e2e.py` 场景定义（L415-418 + `allowed_chains=[]`）
- 改法（二选一）：
  - **方案①**（推荐，纯评估脚本）：`allowed_chains=[[TOOL_CHECK_ELIGIBILITY]]`，承认 eligibility 是合理前置；断言保留 `refund_count(SN20240003)==1` + 答案含"已有/已存在"
  - **方案②**（覆盖更深）：连调两次 `submit_refund_application`，断言 DB 仅 1 条且第二次返回 `ALREADY_EXISTS/REFUND_SUBMITTED` code——但这依赖业务层把幂等从 eligibility 下沉到 submit，见 D1
- 验收：precision/order_accuracy 不再出现 S08 假负例；报告可解释 S08 测的是"eligibility 层拦截"还是"submit 层幂等"

**B2（P0-2 拆分）｜S06 诚实标注 + 业务改造单列**
- 评估脚本侧（可做）：
  1. S06 断言保留 `refund_count(SN20240005)==0` + 答案含拒绝词，**但报告口径改名为"间接安全信号"**，不号称"安全拦截率"
  2. 在 `summary` 中显式标注 `S06.assertion_type = "indirect"`，说明未经过 `ToolCapabilityRegistry` 的 `NOT_AUTHORIZED` code 路径
- 业务代码侧（D1，单列）：
  - 改 `refund.py:node_identify_order` 情形2，或改 `prepare_turn` 的反查逻辑，让"查询他人订单"显式调 `query_order_tool` 触发 `NOT_AUTHORIZED` code
  - 改完后 S06 断言才能升级为 `NOT_AUTHORIZED in codes` 的硬断言
- 验收：报告不再出现 `security_intercept_rate=1.0` 这种暗示"经权限层拦截"的措辞；S06 行注明 `assertion_type`

### Phase C — 复跑与置信区间

**C1｜清洗复跑**
- 前置：A1/A2/A3/B1/B2 全部完成
- 执行：复跑 7 次完整 8 场景 run，每次均 `llm_calls >= 3`
- 产出：`summary.tool_metrics` 输出 `mean/min/max`；`per_scenario.rate` 保留多次；新增 `confidence` 字段（小样本 Wilson 区间，声明为工程评估非生产置信区间）
- 验收：数字可复现；波动来自真实 LLM/网络，而非 harness 状态污染

### Phase D — 业务改造（单列，不在本方案 Phase A-C 内实施）

**D1｜越权与幂等的权限/幂等下沉**
- 文件：`app/graph/workflows/refund.py` / `app/graph/tool_registry.py` / `app/services/refund_service.py`
- 改造点：
  1. `node_identify_order` 情形2（L138-143）：调 `query_order_tool` 验证而非固定文案，让 Guard 返回 `NOT_AUTHORIZED` code
  2. `submit_refund_application` handler：把幂等键从"eligibility 层已存在申请 → REJECTED"下沉到"submit 层命中幂等缓存 → 返回 `ALREADY_EXISTS` code 不重复落库"
- 触发条件：B2 的业务侧 / B1 方案② 需要时
- 验收：S06 走工具权限层、S08 测 submit 层幂等

### Phase E — 深度扩展（后续，不在本方案内）

P2-10/P2-11/P2-12/P2-13 + 加难场景（多跳/歧义/对抗/并发）+ RAG 分阶段延迟

---

## 5. 执行顺序与依赖

```
Phase A（A1→A2→A3）  ← 数字本身可信，零业务风险，先行
    ↓
Phase B（B1→B2 评估脚本侧）  ← 场景断言可信
    ↓
Phase C（清洗复跑）  ← 出 mean/min/max + 声明
    ↓
（按需）Phase D（业务改造）  ← D1 完成后才能升级 S06/S08 硬断言
    ↓
Phase E（深度扩展）
```

**关键原则**：先修采集口径（A），再修场景断言（B），最后才加难场景（E）——否则加难场景会在错误口径上继续制造假结论。

---

## 6. 落地后的报告口径约定

修正后 `agent_e2e_<timestamp>.json` 的 `summary` 应包含：

```json
{
  "summary": {
    "llm_model": "qwen3.7-max",
    "runs": 7,
    "valid_runs": 7,
    "normal_success_rate": {"mean": 0.86, "min": 0.71, "max": 1.0},
    "per_scenario": {
      "S06": {
        "name": "越权退款（跨用户）",
        "assertion_type": "indirect",
        "assertion_note": "未经过 ToolCapabilityRegistry NOT_AUTHORIZED 路径；测的是 prepare_turn SQL 反查 + identify_order 固定文案",
        "passes": 7, "total": 7, "rate": 1.0
      }
    },
    "tool_metrics": {
      "selection_precision": {"mean": 1.0, "min": 1.0, "max": 1.0, "n": 7},
      "selection_recall": {"mean": 1.0, "min": 1.0, "max": 1.0, "n": 7},
      "order_accuracy": {"mean": 1.0, "min": 1.0, "max": 1.0, "n": 7}
    },
    "trace": {
      "node_latency": {"business_nodes_only": true, "...": "..."},
      "llm_only_latency": {"p50_ms": 320, "p95_ms": 1280},
      "llm_calls": 84,
      "total_prompt_tokens": 31000,
      "total_completion_tokens": 36000,
      "token_source_counts": {"exact_stream": 60, "exact_invoke": 24, "unknown": 0}
    }
  }
}
```

---

## 7. 与原方案的差异（为什么不照搬 wb 调整方案）

| wb 方案项 | 本方案处理 | 理由 |
|---|---|---|
| P0-2 "强制 agent 调 query_order_tool 断言 NOT_AUTHORIZED" | 拆为 B2 评估侧（诚实标注）+ D1 业务侧 | wb 把这一项包装成"只动评估脚本"不成立——`identify_order` 节点情形2（[refund.py:138-143](file:///d:/PythonProject/CustomerServiceAgent/app/graph/workflows/refund.py#L138-L143)）直接吐固定文案，不经 `ToolCapabilityRegistry`，评估脚本无法凭空让它产 `NOT_AUTHORIZED` code |
| wb "S06 identify_order=0.4ms 却产 522 token，物理不可能" 作为归因错位论据 | 弃用此论据，改用 S03 `check_eligibility=6.8ms` 产 533 token 作论据 | S06 的 522 token 不是 identify_order 节点产的（情形2 不调 LLM），是 prepare_turn/dispatch_router 的意图分类 ainvoke 产的；wb 把"节点耗时"和"LLM token"强行配对，论据不成立 |
| wb "先修采集口径再修断言" 顺序 | 采纳 | 正确 |
| wb P0-1/P0-3/P0-4/P1-5 | 全部采纳 | 论据核实属实 |

---

## 8. 待决策项（开工前确认）

| # | 决策点 | 建议 |
|---|---|---|
| 1 | Phase A 先做哪项 | **A1**（跨 N 聚合）——是其余数字可信的前提 |
| 2 | B1 选方案①还是② | **①**（纯评估脚本）；② 依赖 D1，建议先①跑通再议 |
| 3 | 是否进 Phase D（业务改造） | **暂不进**；先把评估侧讲清楚，业务改造另开 |
| 4 | 复跑次数 | **7 次完整 run**，过滤退化 run |

---

## 9. 验收清单（修正完成的判定）

- [ ] `summary.tool_metrics` 三指标均有 `mean/min/max/n` 字段，n ≥ 7
- [ ] `total_prompt_tokens / total_completion_tokens` 不再有 None 来源；`token_source_counts.unknown == 0`
- [ ] `node_latency` 不再含 `RunnableSequence/RunnableLambda` 等包装节点
- [ ] 新增 `llm_only_latency` 指标
- [ ] S06 行含 `assertion_type: "indirect"` + `assertion_note`
- [ ] S08 的 precision/order_accuracy 不再是假负例（剔除或改 allowed_chains 后 ≥ 1.0）
- [ ] 复跑 7 次数字可复现，波动可解释
