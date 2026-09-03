from app.evaluation.metrics import answer_clause_metrics, aggregate_metrics, clause_metrics
from app.services.policy_chunks import load_policy_documents
from app.services.policy_retrieval import RetrievedPolicyChunk, _rerank_by_authority


def test_policy_markdown_is_split_at_clause_boundaries():
    documents = load_policy_documents("data/02_category_rules.md")
    clause_ids = [document.metadata["clause_ids"] for document in documents]

    assert len(documents) == 10
    assert clause_ids[0] == []
    assert clause_ids[1] == ["CAT_001"]
    assert "CAT_009" in documents[-1].page_content


def test_clause_metrics_distinguish_primary_rank_and_multi_hop_coverage():
    contexts = [
        {"clause_ids": ["FAQ_011"]},
        {"clause_ids": ["CAT_001"]},
        {"clause_ids": ["VIP_006"]},
    ]
    metrics = clause_metrics(["VIP_006", "CAT_001"], contexts, k=5)

    assert metrics["primary_hit@5"] == 1.0
    assert metrics["any_hit@5"] == 1.0
    assert metrics["clause_recall@5"] == 1.0
    assert metrics["mrr@5"] == 1 / 3


def test_aggregate_metrics_averages_per_question_results():
    summary = aggregate_metrics([
        {"expected_sources": ["A_001"], "retrieved_contexts": [{"clause_ids": ["A_001"]}]},
        {"expected_sources": ["B_001"], "retrieved_contexts": [{"clause_ids": ["X_001"]}]},
    ], k=5)
    assert summary["primary_hit@5"] == 0.5


def test_clause_metrics_preserve_raw_rank_and_do_not_double_count_duplicate_chunks():
    contexts = [
        {"rank": 2, "clause_ids": ["CAT_001"]},
        {"rank": 3, "clause_ids": ["CAT_001"]},
    ]
    metrics = clause_metrics(["CAT_001"], contexts, k=5)

    assert metrics["mrr@5"] == 1 / 2
    assert 0 < metrics["ndcg@5"] < 1.0


def test_authority_rerank_promotes_policy_referenced_by_faq_without_dropping_faq():
    faq = RetrievedPolicyChunk(
        content="FAQ", source="06_faq.md", clause_ids=["FAQ_016"],
        canonical_clause_ids=["QUALITY_004"], source_type="faq", rank=1, distance=0.1,
    )
    authority = RetrievedPolicyChunk(
        content="policy", source="03_quality_return_policy.md", clause_ids=["QUALITY_004"],
        canonical_clause_ids=[], source_type="policy", rank=3, distance=0.2,
    )
    other = RetrievedPolicyChunk(
        content="other", source="03_quality_return_policy.md", clause_ids=["QUALITY_001"],
        canonical_clause_ids=[], source_type="policy", rank=2, distance=0.15,
    )

    ranked = _rerank_by_authority([faq, other, authority], top_k=3)

    assert [chunk.clause_ids[0] for chunk in ranked] == ["QUALITY_004", "FAQ_016", "QUALITY_001"]


def test_answer_clause_metrics_keep_unexpected_citations_as_review_signal():
    metrics = answer_clause_metrics("依据 VIP_001、FAQ_016 和 RETURN_004。", ["VIP_001", "QUALITY_004"])

    assert metrics["answer_primary_source_hit"] is True
    assert metrics["answer_expected_source_recall"] == 0.5
    assert metrics["answer_unexpected_sources"] == ["FAQ_016", "RETURN_004"]


def test_chinese_ragas_prompts_include_domain_few_shot_examples():
    from scripts.evaluate_ragas import _require_ragas
    _require_ragas()  # 安装 RAGAS 的兼容层后再导入内部 prompt 类型。
    from app.evaluation.chinese_ragas_prompts import ChineseNLIStatementPrompt

    prompt = ChineseNLIStatementPrompt().to_string()
    assert "CAT_001" in prompt
    assert "48 小时" in prompt
    assert "JSON Schema" in prompt
