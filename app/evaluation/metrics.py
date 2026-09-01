"""不依赖 LLM 的条款级检索指标。"""
from collections.abc import Iterable
from math import log2
from statistics import mean


def retrieved_clause_ids(contexts: Iterable[dict]) -> list[str]:
    return [clause_id for context in contexts for clause_id in context.get("clause_ids", [])]


def clause_metrics(expected_sources: list[str], contexts: Iterable[dict], *, k: int) -> dict[str, float]:
    """计算主条款命中、覆盖率、MRR 与二值相关性的 nDCG。"""
    expected = set(expected_sources)
    retrieved = retrieved_clause_ids(contexts)[:k]
    hits = [source for source in retrieved if source in expected]
    primary_rank = next((index for index, source in enumerate(retrieved, 1) if source == expected_sources[0]), None)

    dcg = sum((1.0 if source in expected else 0.0) / log2(index + 1)
              for index, source in enumerate(retrieved, 1))
    ideal_length = min(len(expected), k)
    idcg = sum(1.0 / log2(index + 1) for index in range(1, ideal_length + 1))
    return {
        f"primary_hit@{k}": float(bool(primary_rank)),
        f"any_hit@{k}": float(bool(hits)),
        f"clause_recall@{k}": len(set(hits)) / len(expected) if expected else 0.0,
        f"mrr@{k}": 1.0 / primary_rank if primary_rank else 0.0,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }


def aggregate_metrics(items: Iterable[dict], *, k: int) -> dict[str, float]:
    rows = [clause_metrics(item["expected_sources"], item["retrieved_contexts"], k=k) for item in items]
    return {key: mean(row[key] for row in rows) for key in rows[0]} if rows else {}
