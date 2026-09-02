"""不依赖 LLM 的条款级检索指标。"""
from collections.abc import Iterable
from math import log2
import re
from statistics import mean


CLAUSE_ID_PATTERN = re.compile(r"\b(?:RETURN|CAT|QUALITY|VIP|SHIP|FAQ)_\d{3}\b")


def retrieved_clause_ids(contexts: Iterable[dict]) -> list[str]:
    return [clause_id for context in contexts for clause_id in context.get("clause_ids", [])]


def clause_metrics(expected_sources: list[str], contexts: Iterable[dict], *, k: int) -> dict[str, float]:
    """计算主条款命中、覆盖率、MRR 与二值相关性的 nDCG。"""
    expected = set(expected_sources)
    ordered_contexts = sorted(contexts, key=lambda context: context.get("rank", float("inf")))
    ranked_contexts = [
        {**context, "rank": context.get("rank", index)}
        for index, context in enumerate(ordered_contexts, 1)
    ]
    ranked_contexts = [context for context in ranked_contexts if context["rank"] <= k]
    retrieved = retrieved_clause_ids(ranked_contexts)
    hits = [source for source in retrieved if source in expected]
    primary_rank = next((context["rank"] for context in ranked_contexts
                         if expected_sources[0] in context.get("clause_ids", [])), None)

    seen_relevant: set[str] = set()
    dcg = 0.0
    for context in ranked_contexts:
        novel_relevant = (set(context.get("clause_ids", [])) & expected) - seen_relevant
        if novel_relevant:
            # 一个 chunk 最多贡献一次相关性，避免长条款多切片造成 nDCG > 1。
            dcg += 1.0 / log2(context["rank"] + 1)
            seen_relevant.update(novel_relevant)
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


def aggregate_by(items: Iterable[dict], group_key: str, *, k: int) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item[group_key], []).append(item)
    return {group: aggregate_metrics(group_items, k=k) for group, group_items in groups.items()}


def answer_clause_metrics(answer: str, expected_sources: list[str]) -> dict:
    """衡量回答对权威条款的引用；unexpected 仅供审查，不直接视为事实错误。"""
    cited = sorted(set(CLAUSE_ID_PATTERN.findall(answer)))
    expected = set(expected_sources)
    matched = sorted(set(cited) & expected)
    return {
        "cited_sources": cited,
        "answer_primary_source_hit": bool(expected_sources and expected_sources[0] in cited),
        "answer_expected_source_recall": len(matched) / len(expected) if expected else 0.0,
        "answer_expected_source_precision": len(matched) / len(cited) if cited else 0.0,
        "answer_unexpected_sources": sorted(set(cited) - expected),
    }


def cosine_similarity(first: list[float], second: list[float]) -> float | None:
    denominator = sum(value * value for value in first) ** 0.5 * sum(value * value for value in second) ** 0.5
    return sum(left * right for left, right in zip(first, second)) / denominator if denominator else None
