"""验证知识库是否已按条款级元数据完成入库。"""
import asyncio
from pathlib import Path
import sys
from sqlmodel import select

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.database import async_session_maker
from app.models.knowledge import KnowledgeChunk


async def main() -> None:
    async with async_session_maker() as session:
        chunks = (await session.exec(select(KnowledgeChunk))).all()
    clause_chunks = [chunk for chunk in chunks if (chunk.meta_data or {}).get("clause_ids")]
    print(f"chunk_count={len(chunks)}")
    print(f"clause_metadata_count={len(clause_chunks)}")
    print(f"sample={[(chunk.source, chunk.meta_data.get('clause_ids')) for chunk in clause_chunks[:5]]}")


if __name__ == "__main__":
    asyncio.run(main())
