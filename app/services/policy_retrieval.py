"""线上节点与评估脚本共用的政策检索能力。"""
from dataclasses import dataclass
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
    """检索并保留评估所需的来源、条款、排序及距离信息。"""
    query_vector = await embedding_model.aembed_query(question)
    async with async_session_maker() as session:
        distance_col = KnowledgeChunk.embedding.cosine_distance(query_vector).label("distance")  # type: ignore
        result = await session.exec(
            select(KnowledgeChunk, distance_col)
            .where(KnowledgeChunk.is_active)  # type: ignore
            .order_by(distance_col)
            .limit(top_k)
        )
        rows = result.all()

    retrieved: list[RetrievedPolicyChunk] = []
    for raw_rank, (chunk, distance) in enumerate(rows, 1):
        if distance < similarity_threshold:
            metadata: dict[str, Any] = chunk.meta_data or {}
            retrieved.append(RetrievedPolicyChunk(
                content=chunk.content,
                source=chunk.source,
                clause_ids=list(metadata.get("clause_ids", [])),
                canonical_clause_ids=list(metadata.get("canonical_clause_ids", [])),
                source_type=str(metadata.get("source_type", "policy")),
                # 保留数据库原始检索位次；阈值过滤不能改变 MRR/nDCG 的排名语义。
                rank=raw_rank,
                distance=float(distance),
            ))
    return retrieved


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
