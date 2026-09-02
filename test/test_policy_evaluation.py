from app.evaluation.metrics import aggregate_metrics, clause_metrics
from app.services.policy_chunks import load_policy_documents


def test_policy_markdown_is_split_at_clause_boundaries():
    documents = load_policy_documents("data/02_category_rules.md")
    clause_ids = [document.metadata["clause_ids"] for document in documents]

    assert len(documents) == 9
    assert clause_ids[0] == ["CAT_001"]
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
