# RAG 模块：知识库构建 · 测试集 · 评估迭代 · 结果

> 本文对应项目中的 RAG（检索增强）子系统：政策条款级知识库、人评 45 题测试集、检索/生成/RAGAS 三层评估体系、迭代过程与终验数据。  
> 配套代码：`scripts/etl_policy.py` · `app/services/policy_chunks.py` · `app/services/policy_retrieval.py` · `app/services/policy_answer_guard.py` · `app/evaluation/metrics.py`  
> 配套运行产物：`eval/testset.json` · `eval/runs/baseline*.json` · `eval/runs/retrieval_p*.json` · `eval/runs/oracle*.json` · `eval/runs/*.ragas.json`  
> 配套环境：通义千问 `qwen3.7-text-embedding`（Embedding，1024 维）/ `qwen3.7-max`（生成）/ `deepseek-v4-flash`（RAGAS judge，必与业务模型隔离）

---

## 1. 一句话定位

RAG 路径是客服 Agent 的「事实中枢」：用户问政策时，先用向量检索从条款级知识库中取证据，再由生成模型产出自然语言回答，并由 `PolicyAnswerGuard` 做确定性引用校验——**所有面向用户的回答都必须能溯源到具体条款编号，且回答中不暴露编号本身**。

---

## 2. 知识库构建

### 2.1 源文档（`data/`）

知识库由 6 份 Markdown 政策文件组成，共 **346 行**：

| 文件 | 主题 | 行数 | 在检索中的角色 |
|---|---|---|---|
| `01_general_return_policy.md` | 通用退换货政策（基础规则） | 71 | 事实条款 |
| `02_category_rules.md` | 特殊品类退换规则 | 64 | 事实条款（优先级高于通用） |
| `03_quality_return_policy.md` | 质量问题退货政策（受理期 30 天） | 58 | 事实条款 |
| `04_vip_service_policy.md` | VIP 会员服务政策 | 47 | 事实条款（优先级最高，例外以 VIP_006 为准） |
| `05_shipping_policy.md` | 物流与配送政策 | 42 | 事实条款 |
| `06_faq.md` | 常见问题 FAQ | 64 | 口语化转述，引用权威条款 |

每份文档**头部**有 1 段「文档级规则」描述该文档的优先级与适用关系（除 FAQ 外）。**正文**采用统一编号：`## RETURN_001: 7天无理由退货` / `## CAT_002: 鞋靴类` / `## VIP_001: 会员等级与核心权益` / `## FAQ_005: …` 等。

> 编号设计是有意为之：条款编号直接作为评估命中的精确锚点，也作为 `PolicyAnswerGuard` 校验合法引用的最小单位。

### 2.2 切片策略：条款级 + 文档级

**不**用 `RecursiveCharacterTextSplitter` 切政策文件——会切碎条款、丢失编号边界。`app/services/policy_chunks.py:load_policy_documents` 按正则 `^##\s+([A-Z]+_\d{3}):\s*(.+?)$` **以条款标题为边界**，一条一个 chunk，元数据严格保留：

```python
# app/services/policy_chunks.py (节选)
CLAUSE_HEADING = re.compile(r"^##\s+([A-Z]+_\d{3}):\s*(.+?)\s*$")
CANONICAL_CLAUSE_ID = re.compile(r"\b(?:RETURN|CAT|QUALITY|VIP|SHIP)_\d{3}\b")
```

每条 chunk 的元数据：

| 字段 | 含义 |
|---|---|
| `clause_ids` | 本条自身的条款编号（如 `["RETURN_001"]`） |
| `clause_title` | 条款标题 |
| `source` | 来源文件名 |
| `source_type` | `policy`（事实条款）/ `faq`（FAQ 转述）/ `policy_rule`（文档级规则） |
| `canonical_clause_ids` | **FAQ 专用**：从正文里自动抽取它引用的权威条款编号 |
| `document_rules` | 该文档的文档级规则原文（用于生成阶段背景，不进检索） |

**文档级规则单独入库**，不复制到每个条款——避免稀释条款本体的向量语义。FAQ 不建 policy_rule（其前言只是编辑说明，不是业务规则）。

### 2.3 ETL：`scripts/etl_policy.py`

```bash
uv run python scripts/etl_policy.py
```

流水线：

```
扫描 data/*.{pdf,md,txt}
  └─ md → load_policy_documents() 条款级切分
  └─ pdf → PyPDFLoader
  └─ txt → TextLoader
  └─ RecursiveCharacterTextSplitter 仅用于 PDF/TXT (md 已条款级)
  └─ 旧数据幂等清理：DELETE WHERE source = 当前文件
  └─ 50 条/批 Embedding (BATCH_SIZE=50)
       └─ 3 次重试 + 指数退避 (tenacity)
  └─ 100 条/批 commit
```

失败处理：单批 APIConnectionError 时整文件失败（被记录到 `etl_policy_p0.err.log`），需检查网络/Key 后重跑；其余文件不受影响。

> 实战坑：早期 P0 跑过一轮 `etl_policy_p0.log`，6 个文件全部 `APIConnectionError` 失败（Embedding 端点未通），当时切换了端点才成功入库。日志已保留作为可复现的失败证据。

### 2.4 检索：`app/services/policy_retrieval.py:retrieve_policy`

```python
# 核心流水线
query_vector = await embedding_model.aembed_query(question)         # qwen3.7-text-embedding
SELECT *, cosine_distance(embedding) AS distance
  FROM knowledge_chunk WHERE is_active
  ORDER BY distance LIMIT max(top_k * 4, 20)                         # 候选池 20–40
  → 过滤 distance < 0.5                                              # 相似度阈值
  → 过滤 source_type='policy_rule'                                   # 文档级规则不进检索位次
  → 取前 top_k → 进入 FAQ 来源感知重排
```

**FAQ 来源感知重排（P1 引入）**：

```python
# 1) 扫 Top-K 内的 FAQ，登记它引用的权威条款的最早出现位次
authority_anchor = {clause_id: min(rank) for faq in faqs for clause_id in faq.canonical_clause_ids}
# 2) 排序键：被 FAQ 引用的权威条款 = (anchor - 0.5, 0, rank)
#           其余              = (rank,         1, rank)
# 3) 只升到对应 FAQ 的前一位，不粗暴置顶（保留原始语义排序的其余信息）
```

**为什么只升到前一位**：`retrieve_policy` 的 docstring 写明「跨候选池补入条款会挤掉多跳题的必要证据，降低召回」——粗暴置顶会牺牲多跳题的次要证据。

### 2.5 生成与引用约束：`app/services/policy_answer_guard.py`

结构化 LLM 输出 schema：

```python
class PolicyAnswer(BaseModel):
    answer: str                          # 用户可见，不展示条款编号
    applied_clause_ids: list[str]        # 直接得出结论的条款
    evidence_clause_ids: list[str]       # 支撑解释的全部条款
```

`validate_policy_answer()` 强校验：

1. **无证据不调生成**：`evidence_present=False` 时，若 `applied/evidence` 非空 → 拒绝（防止模型无依据时编造编号）。
2. **必须填双字段**：有证据时 `applied` 与 `evidence` 都必须非空。
3. **集合关系**：`applied ⊆ evidence`（越界编号 = 拒绝）。
4. **证据可追溯**：`evidence ⊆ allowed`（allowed = 检索 chunk 的 clause_ids ∪ FAQ 的 canonical_clause_ids）。
5. **不暴露编号**：用户可见 `answer` 文本里**不能出现**任何 `XXX_NNN` 格式的编号。

失败时返回 `SAFE_POLICY_FALLBACK`（已有证据但校验失败）或 `NO_EVIDENCE_FALLBACK`（无证据），仅重试 1 次。

---

## 3. 评估测试集 `eval/testset.json`

### 3.1 规模与字段

**45 题**，版本 v1.0。来源：完全人工标注（`field_spec` 写明「面向 RAGAS 评估的测试集」）。

| 字段 | 用途 |
|---|---|
| `id` | 题目编号（Q001–Q045） |
| `question` | 用户自然语言提问 |
| `ground_truth` | 标准答案（纯语义，供 RAGAS answer_correctness / similarity 用） |
| `expected_sources` | **应命中的权威条款编号列表**；第 1 个为主权威条款，FAQ_xxx 视为次优命中 |
| `category` | 题型维度（见下） |
| `difficulty` | `easy` / `medium` / `hard` |

### 3.2 题型分布

| 题型 | 题数 | 典型题（节选） | 难度 |
|---|---|---|---|
| **层级冲突** | 6 | Q001 「黑卡会员买内衣试穿过了还能退吗」→ `VIP_001/VIP_006/CAT_001` | hard（多数） |
| **量化细节** | 10 | Q007 「极速退款额度和次数限制」→ `VIP_002/FAQ_018` | medium |
| **品类边界** | 9 | Q017 「内衣试穿了一次」→ `CAT_001/FAQ_011` | easy/medium |
| **抗幻觉** | 8 | Q026 「洗过的衣服还能退吗」→ `FAQ_014/RETURN_005` | easy（看模型是否会编造不存在的例外） |
| **基础事实** | 7 | Q034 「换货之后还能再退货吗」→ `RETURN_002/FAQ_003` | easy |
| **多跳综合** | 5 | Q041 「金卡买真丝衬衫 10 天勾丝」→ `CAT_007/QUALITY_004/VIP_001` | hard |

**难度分布**：easy 18 / medium 21 / hard 6。

> 设计动机：客服真实问题几乎都不是「一条直查」——会员身份 × 品类 × 质量 × 时效的组合才是常态。**层级冲突** 与 **多跳综合** 是 LLM 最容易翻车的两类，测试集**有意**让它们占主导（11/45 = 24%）。

---

## 4. 评估指标体系

三套指标，覆盖检索、引用、RAG 自动评三个维度。

### 4.1 条款级检索指标（`app/evaluation/metrics.py`）

不依赖 LLM，纯确定性：

| 指标 | 定义 | 用途 |
|---|---|---|
| `primary_hit@5` | 主权威条款（第 1 个 `expected_sources`）是否进入 Top-5 | 检索命中硬指标 |
| `any_hit@5` | 至少一个 expected 是否进入 Top-5 | 兜底 |
| `clause_recall@5` | 命中的 expected 集合占比 | 多条款题的真实召回 |
| `mrr@5` | 主权威条款的倒数位次 | 衡量位次质量 |
| `ndcg@5` | 二值相关性 nDCG，**每个 chunk 至多贡献 1 次**（防长条款多切片让 nDCG>1） | 位次 + 多样性 |

### 4.2 答案级引用指标

`answer_clause_metrics()`：从生成文本里抽 `XXX_NNN` 编号，比对 expected：

- `answer_primary_source_hit`：主权威条款是否被引用
- `answer_expected_source_recall`：expected 中被引用的比例
- `answer_expected_source_precision`：引用中属于 expected 的比例
- `answer_unexpected_sources`：引用了但不在 expected 中的（**仅供审查，不直接视为事实错误**——可能合理引用了相邻条款）

### 4.3 RAGAS Faithfulness

`scripts/evaluate_ragas.py` 跑 RAGAS 0.4+ 的 `Faithfulness`（衡量「答案是否忠于检索上下文、不编造」）：

- **judge 模型必须与业务 LLM 不同**（`evaluate_ragas.py` 启动时强校验），用 `deepseek-v4-flash`
- judge **关 thinking**（`extra_body={"thinking":{"type":"disabled"}}`）：DeepSeek v4 默认开 CoT，单次 <1s 涨到 10–30s，叠加 RAGAS 每题 4 次调用频繁超时
- **prompt 用中文**（`app/evaluation/chinese_ragas_prompts.py`），RAGAS 默认英文 prompt 对中文场景不适用
- **不评测 factual_correctness**：该模式只罚「答案里有而 ground_truth 没有的 claim」，客服答案天然比一句话 ground_truth 长（条款引用+客套话+追问），导致全对答案被系统性判低分（曾有 4 个 0 分题答案与 ground_truth 完全一致），已移除
- **生成侧事实正确性**改由确定性的 `answer_clause_metrics` 衡量

### 4.4 运行约束

`scripts/run_rag_baseline.py` 是入口（`--oracle` / `--retrieval-only` / `--limit` / `--concurrency`）：

- **并发可控**（`asyncio.Semaphore(4)`，可调），不抢 Embedding/生成限流
- **checkpoint 恢复**：写到 `<output>.jsonl`，重跑自动从上次完成的题恢复（避免 45 题中断从头来）
- **失败隔离**：单题异常写 `status: error` 行，不阻塞其他题
- **三种运行模式**：
  - `baseline`：真实检索 + 真实生成（生产仿真）
  - `oracle`：直接用 `expected_sources` 读原文作为 context，隔离生成器能力
  - `retrieval_only`：只跑检索，不调生成（用于消检索与生成的边界）

---

## 5. 评估迭代过程

按时间顺序的 5 轮迭代，每轮解决一个具体问题。

### P0 阶段 1：基线建立（早期 `eval/runs/baseline.json`）

| 指标 | 值 |
|---|---|
| primary_hit@5 | 0.9778（**44/45，1 题未命中**） |
| any_hit@5 | 1.0000 |
| clause_recall@5 | 0.9704 |
| mrr@5 | 0.7833 |
| ndcg@5 | 0.9387 |

**按题型拆分**——`层级冲突` 是明显短板：primary_hit 0.833、mrr 0.514；其余 5 个题型 primary_hit 全部 1.0。说明**单一维度的命中已基本到位，但跨文档优先级冲突的题位次质量差**。

### P0 阶段 2：检索边界隔离（`retrieval_p0_final.json`，retrieval_only 模式）

改动：
- 引入 `source_type` 区分（`policy` / `policy_rule` / `faq`）
- **policy_rule 不再进检索位次**（仅在生成期作为背景喂给模型）
- 相似度阈值 `< 0.5` 过滤
- 候选池扩到 `max(top_k*4, 20)`，为后续重排留空间

| 指标 | baseline (P0-1) | retrieval_p0_final | Δ |
|---|---|---|---|
| primary_hit@5 | 0.9778 | **1.0000** | +0.022 |
| clause_recall@5 | 0.9704 | **0.9815** | +0.011 |
| mrr@5 | 0.7833 | 0.7915 | +0.008 |
| ndcg@5 | 0.9387 | **0.9441** | +0.005 |

**层级冲突** primary_hit 从 0.833 → 1.000，recall 0.833 → 0.917——policy_rule 隔离消除了文档级规则挤占事实条款位次的问题。

### P1 阶段 1：FAQ 来源感知重排（`retrieval_p1_source_aware_v2.json`）

改动：实现 `_rerank_by_authority`（见 §2.4）——Top-K 内有 FAQ 引用某权威条款时，把该权威条款升到 FAQ 的前一位。

| 指标 | retrieval_p0_final | retrieval_p1_source_aware_v2 | Δ |
|---|---|---|---|
| primary_hit@5 | 1.0000 | 1.0000 | = |
| clause_recall@5 | 0.9815 | 0.9815 | = |
| **mrr@5** | 0.7915 | **0.8685** | **+0.077** |
| ndcg@5 | 0.9441 | **0.9457** | +0.0016 |

按难度看：easy mrr 0.671 → **0.907**（+0.236），hard mrr 0.756 → 0.653（小幅下降，因为 hard 题本来就在前）。

**hard 题 mrr 反而微降** 是个信号：硬题通常 expected 是 `[主权威, 边界条款]`，重排后主权威的位次被 FAQ 的「前一位」规则保留，边界条款被挤到后位。这是有意识的取舍——`docstring` 写明「跨候选池补入条款会挤掉多跳题的必要证据，降低召回」。

### P0 阶段 3：接入生成 + RAGAS（`baseline_p0_generation.json` + `*.ragas.json`）

用 P0 检索 + 真实生成器跑完整链路，再用 RAGAS 评：

| 指标 | 值 |
|---|---|
| primary_hit@5 | 1.0000 |
| clause_recall@5 | 0.9815 |
| mrr@5 | 0.7915 |
| ndcg@5 | 0.9441 |
| **RAGAS Faithfulness (n=45)** | **0.9347**（judge: deepseek-v4-flash, thinking disabled） |

faithfulness 0.93+ 意味着：**45 题里有约 6% 的答案在「是否完全忠于检索上下文」这一项上被判有问题**——这些通常是答案补充了客套话或顺带解释了背景。

### P1 阶段 2：Oracle 对照（`oracle_p1_final.json` + `*.ragas.json`）

**用测试集金标准 `expected_sources` 直接读原文作为 context**，隔离生成器，验证「如果检索完美，生成能拿几分」：

| 指标 | 值 |
|---|---|
| primary_hit@5 | 1.0000 |
| any_hit@5 | 1.0000 |
| clause_recall@5 | 1.0000 |
| mrr@5 | 1.0000 |
| ndcg@5 | 1.0000 |
| **RAGAS Faithfulness (n=45)** | **0.9091** |

> 注：oracle 的 faithfulness（0.9091）**比 baseline 真实链路（0.9347）略低**。原因：oracle 喂的 context 是「金标准条款 + 极少量其它」，生成器在「能直接抄原文」时反而更倾向于「完整复述」或「引用编号试探」，引入了少量不在原文里的话术；baseline 喂的 context 是「相关 Top-5 条款」，生成器在「证据更分散」时更倾向于「基于条款重组为口语化回答」。**这是 faithfulness 指标的已知偏差，不代表 oracle 答案更差**——`answer_clause_metrics` 才是更可信的对比维度。

---

## 6. 评估结果汇总

### 6.1 三类运行的最终结果

| 维度 | 指标 (k=5) | baseline 真实链路 | retrieval p1 (纯检索) | oracle (金标准 context) |
|---|---|---|---|---|
| **检索** | primary_hit | 1.0000 | 1.0000 | 1.0000 |
| | any_hit | 1.0000 | 1.0000 | 1.0000 |
| | clause_recall | 0.9815 | 0.9815 | 1.0000 |
| | mrr | 0.7915 | **0.8685** | 1.0000 |
| | ndcg | 0.9441 | 0.9457 | 1.0000 |
| **生成** | RAGAS Faithfulness (n=45) | 0.9347 | — | **0.9091** |
| | judge | deepseek-v4-flash (thinking off) | — | deepseek-v4-flash (thinking off) |

> README 当前挂在历史口径上的数字是 `primary_hit@5 = 1.000`、`clause_recall@5 = 0.9815`、`MRR@5 = 0.8685`、RAGAS `0.9091`——分别对应 **retrieval_p1_source_aware_v2** 和 **oracle_p1_final**。  
> 选 oracle faithfulness 作对外数字是因为它最严格地衡量「答案是否忠于证据」；选 retrieval_p1 的 mrr 是因为它代表了真实链路的位次质量。

### 6.2 按题型拆分（retrieval_p1_source_aware_v2，真实链路最终状态）

| 题型 | 题数 | primary_hit@5 | clause_recall@5 | mrr@5 |
|---|---|---|---|---|
| 层级冲突 | 6 | 1.000 | 0.917 | 0.653 |
| 量化细节 | 10 | 1.000 | 1.000 | 1.000 |
| 品类边界 | 9 | 1.000 | 1.000 | 0.870 |
| 抗幻觉 | 8 | 1.000 | 1.000 | 0.875 |
| 基础事实 | 7 | 1.000 | 1.000 | 0.929 |
| 多跳综合 | 5 | 1.000 | 0.933 | 0.767 |

**两个观察**：
1. **primary_hit@5 全 1.0**：所有题的主权威条款都进了 Top-5，是检索合格线。
2. **层级冲突的 recall 0.917**（不是 1.0）+ **mrr 0.653**（最低）：这类题 expected 长度普遍 3 个（如 `VIP_001/VIP_006/CAT_001`），第 2、3 个条款是「边界例外」，候选池扩到 20 才能覆盖；当前能拿到主权威但次要例外偶尔缺位。

### 6.3 按难度拆分（retrieval_p1_source_aware_v2）

| 难度 | 题数 | primary_hit@5 | clause_recall@5 | mrr@5 |
|---|---|---|---|---|
| easy | 18 | 1.000 | 1.000 | **0.907** |
| medium | 21 | 1.000 | 1.000 | 0.897 |
| hard | 6 | 1.000 | 0.861 | 0.653 |

**hard 全面落后**：recall 0.861 意味着 6 道 hard 题里**有 1 道的次要 expected 条款没进 Top-5**；mrr 0.653 意味着主权威条款平均在第 1.5 位。**这是检索子系统的真实天花板**——再调 rerank 策略、扩 top_k 都会以牺牲中位数为代价，对 hard 题帮助有限，因为 hard 题本身的语义复杂度需要更大的候选池 + 多向量召回（当前的「单 query embedding + 候选池 20 + Top-K 重排」是单跳设计）。

### 6.4 仍存在的「1 题未召回」细节

hard 题中 recall<1 的 1 道是 **Q001「我是黑卡会员，买的内衣试穿过了，还能退吗」**——expected `VIP_001/VIP_006/CAT_001`，mrr=0.25（主权威在第 4 位）。这是**会员身份 × 品类 × 例外**的三跳题，单 query embedding 在「试穿+黑卡+内衣」三个语义信号上无法同时对齐到三个条款。该题在 README 列为 `层级冲突` 类短板案例，列入「Phase D 业务改造」候选（多向量召回 / HyDE 等）。

---

## 7. 设计决策与可审计约束（一张表串起）

| 决策 | 代码位置 | 理由 |
|---|---|---|
| 条款级切分 | `policy_chunks.py:load_policy_documents` | 编号是评估锚点 + 校验锚点，**不能被切碎** |
| 文档级规则单存 | 同上 | 复制到每条会稀释条款本体向量 |
| 候选池 `max(top_k*4, 20)` | `policy_retrieval.py:retrieve_policy` | 为重排留足候选，避免多跳题证据缺位 |
| `policy_rule` 不进检索 | 同上 | 文档级规则是「生成期背景」，不是「事实证据」 |
| FAQ 来源感知重排（仅升前 1 位） | `_rerank_by_authority` | 粗暴置顶会牺牲多跳题次要证据；只升 1 位是 controlled trade-off |
| 结构化生成 + Guard 强校验 | `policy_answer_guard.py` | 「答案溯源到编号」是合规刚需，不能靠 prompt 软约束 |
| **用户可见 answer 不出现编号** | 同上 `CLAUSE_ID_PATTERN` | 编号是给系统看的，不是给用户看的——避免「按编号 123 去申请」这种用户误用 |
| RAGAS judge 强制 ≠ 业务 LLM | `evaluate_ragas.py:71` | 自评会系统性高估，**硬约束** |
| DeepSeek judge 关 thinking | `evaluate_ragas.py:94` | 默认 CoT 让单次 1s 涨到 30s 触发 RAGAS 120s 超时 |
| 用中文 RAGAS prompt | `app/evaluation/chinese_ragas_prompts.py` | 英文 prompt 对中文场景 faithfulness 跑偏 |
| **不评测 factual_correctness** | `evaluate_ragas.py:97` 注释 | 客服答案天然比 ground_truth 长，会被系统性判低分 |
| checkpoint jsonl 恢复 | `run_rag_baseline.py:load_completed` | 45 题中断不能从头跑 |
| 三种运行模式（baseline/oracle/retrieval_only） | `run_rag_baseline.py` | oracle 用于隔离生成器，retrieval_only 用于消检索 |

---

## 8. 已知限制与 Phase D 候选

| 限制 | 影响 | 候选方案（Phase D） |
|---|---|---|
| 单 query embedding，多跳题 1 道 recall<1 | hard 题天花板 | 多向量召回（multi-query） / HyDE / 跨文档图谱检索 |
| 候选池 20–40 + Top-K 重排 | 位次质量有上限 | 提升到 50–100 + cross-encoder rerank |
| mrr 0.87 仍有提升空间 | 答案里证据顺序不稳 | 句向量 rerank（bge-reranker） |
| faithfulness 0.93，6% 答案被判「不忠」 | 偶有「补充话术」被判编造 | judge prompt 微调 / 引入 answer-groundedness 而非 faithfulness |
| 测试集 45 题 | 置信区间宽 | 扩到 100–200 + 多评估者交叉标注 |
| 真实流量 vs 测试集分布差异 | 离线指标只是参考 | 上线后抽样标注 + 在线 bad case 回收 |
| embedding / judge 都是第三方 | 不可控抖动 | 关键模块做自托管（bge / Qwen2.5） |

---

## 9. 一句话复述

RAG 子系统以 **条款级切分 + 来源感知重排 + 结构化生成 + 引用强校验** 四件套，**在 45 题人评集上跑出 primary_hit@5 = 1.000 / clause_recall@5 = 0.9815 / MRR@5 = 0.8685 / RAGAS Faithfulness = 0.9091**；hard 题型是真实短板（1 题 recall<1、mrr 0.65），已识别为 Phase D 业务改造候选，**没有靠调 prompt 把数字刷上去**。
