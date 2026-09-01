"""Markdown 政策文档的条款级切分。"""
from pathlib import Path
import re

from langchain_core.documents import Document


CLAUSE_HEADING = re.compile(r"^##\s+([A-Z]+_\d{3}):\s*(.+?)\s*$")


def load_policy_documents(file_path: str) -> list[Document]:
    """按 ``## POLICY_001: 标题`` 切分 Markdown，并保留条款编号元数据。"""
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    documents: list[Document] = []
    current_id: str | None = None
    current_title = ""
    current_lines: list[str] = []

    def flush() -> None:
        if current_id and (content := "\n".join(current_lines).strip()):
            documents.append(Document(
                page_content=f"## {current_id}: {current_title}\n\n{content}",
                metadata={
                    "source": path.name,
                    "clause_ids": [current_id],
                    "clause_title": current_title,
                },
            ))

    for line in lines:
        match = CLAUSE_HEADING.match(line)
        if match:
            flush()
            current_id, current_title = match.groups()
            current_lines = []
        elif current_id:
            current_lines.append(line)
    flush()
    return documents
