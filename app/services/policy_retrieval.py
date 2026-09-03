"""线上节点与评估脚本共用的政策检索能力。"""
from dataclasses import dataclass, replace
from typing import Any, List

import httpx
from langchain_core.embeddings import Embeddings
from app.core.config import settings
from app.core.database import async_session_maker
from app.models.knowledge import KnowledgeChunk
from sqlmodel import select


@dataclass(frozen=True)
class RetrievedPolicyChunk:
    content: str
    source: str
    clause_ids: list[str]
    canonical_clause_ids: list[str]
    source_type: str
    rank: int
    distance: float


class QwenEmbeddings(Embeddings):
    """通义千问 Embedding API 的异步适配器。"""

    def __init__(self, base_url: str, api_key: str, model: str, dimensions: int):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("请使用异步方法 aembed_documents")

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("请使用异步方法 aembed_query")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "input": texts, "dimensions": self.dimensions},
                timeout=30.0,
            )
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([text]))[0]


embedding_model = QwenEmbeddings(
    base_url=settings.OPENAI_BASE_URL,
    api_key=settings.OPENAI_API_KEY,
    model=settings.EMBEDDING_MODEL,
    dimensions=settings.EMBEDDING_DIM,
)


async def retrieve_policy(question: str, *, top_k: int = 5, similarity_threshold: float = 0.5) -> list[RetrievedPolicyChunk]:
    """检索并在原始 Top-K 内进行 FAQ 来源感知重排。"""
    query_vector = await embedding_model.aembed_query(question)
    async with async_session_maker() as session:
        distance_col = KnowledgeChunk.embedding.cosine_distance(query_vector).label("distance")  # type: ignore
        result = await session.exec(
            select(KnowledgeChunk, distance_col)
            .where(KnowledgeChunk.is_active)  # type: ignore
            .order_by(distance_col)
            # 保留扩展候选池，为后续 rerank 使用；本阶段不以额外权威条款替换 Top-K。
            .limit(max(top_k * 4, 20))
        )
        rows = result.all()

    retrieved: list[RetrievedPolicyChunk] = []
    for raw_rank, (chunk, distance) in enumerate(rows, 1):
        metadata: dict[str, Any] = chunk.meta_data or {}
        # 文档级规则只服务生成期的决策背景，不能挤占事实条款的检索位次。
        if metadata.get("source_type") == "policy_rule":
            continue
        if distance < similarity_threshold:
            retrieved.append(RetrievedPolicyChunk(
                content=chunk.content,
                source=chunk.source,
                clause_ids=list(metadata.get("clause_ids", [])),
                canonical_clause_ids=list(metadata.get("canonical_clause_ids", [])),
                source_type=str(metadata.get("source_type", "policy")),
                # policy_rule 被明确排除，排名是事实条款候选池中的原始位次。
                rank=len(retrieved) + 1,
                distance=float(distance),
            ))
    # 仅在原始 Top-K 内重排：跨候选池补入条款会挤掉多跳题的必要证据，降低召回。
    return _rerank_by_authority(retrieved[:top_k], top_k=top_k)


def _rerank_by_authority(
    candidates: list[RetrievedPolicyChunk], *, top_k: int,
) -> list[RetrievedPolicyChunk]:
    """将 Top-K 内 FAQ 明示引用的权威条款置于该 FAQ 之前，保留 FAQ 作为口语化证据。"""
    authority_anchor: dict[str, int] = {}
    for candidate in candidates:
        if candidate.source_type != "faq":
            continue
        for clause_id in candidate.canonical_clause_ids:
            authority_anchor[clause_id] = min(authority_anchor.get(clause_id, candidate.rank), candidate.rank)

    def ranking_key(candidate: RetrievedPolicyChunk) -> tuple[float, int, int]:
        is_referenced_authority = (
            candidate.source_type == "policy"
            and any(clause_id in authority_anchor for clause_id in candidate.clause_ids)
        )
        if is_referenced_authority:
            # 只提升到对应 FAQ 的前一位，而非粗暴置顶，保留原始语义排序的其余信息。
            anchor = min(authority_anchor[clause_id] for clause_id in candidate.clause_ids if clause_id in authority_anchor)
            return (anchor - 0.5, 0, candidate.rank)
        return (float(candidate.rank), 1, candidate.rank)

    ranked = sorted(candidates, key=ranking_key)[:top_k]
    return [replace(chunk, rank=index) for index, chunk in enumerate(ranked, 1)]


async def load_policy_rules() -> list[str]:
    """读取政策生成需要的全局优先级背景；不混入检索候选。"""
    async with async_session_maker() as session:
        rows = (await session.exec(select(KnowledgeChunk).where(KnowledgeChunk.is_active))).all()  # type: ignore
    rules = [
        chunk.content for chunk in rows
        if (chunk.meta_data or {}).get("source_type") == "policy_rule"
    ]
    return sorted(rules)


async def load_oracle_contexts(clause_ids: list[str]) -> list[RetrievedPolicyChunk]:
    """按测试集金标准读取条款原文，用于隔离生成器能力。"""
    async with async_session_maker() as session:
        rows = (await session.exec(select(KnowledgeChunk).where(KnowledgeChunk.is_active))).all()  # type: ignore

    by_clause: dict[str, list[KnowledgeChunk]] = {}
    for chunk in rows:
        for clause_id in (chunk.meta_data or {}).get("clause_ids", []):
            by_clause.setdefault(clause_id, []).append(chunk)

    contexts: list[RetrievedPolicyChunk] = []
    for clause_id in clause_ids:
        for chunk in by_clause.get(clause_id, []):
            contexts.append(RetrievedPolicyChunk(
                content=chunk.content,
                source=chunk.source,
                clause_ids=[clause_id],
                canonical_clause_ids=list((chunk.meta_data or {}).get("canonical_clause_ids", [])),
                source_type=str((chunk.meta_data or {}).get("source_type", "policy")),
                rank=len(contexts) + 1,
                distance=0.0,
            ))
    return contexts
