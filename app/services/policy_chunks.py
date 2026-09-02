"""Markdown 政策文档的条款级切分。"""
from pathlib import Path
import re

from langchain_core.documents import Document


CLAUSE_HEADING = re.compile(r"^##\s+([A-Z]+_\d{3}):\s*(.+?)\s*$")
# FAQ 是面向用户的转述；其中显式写出的政策编号可作为其权威依据。
CANONICAL_CLAUSE_ID = re.compile(r"\b(?:RETURN|CAT|QUALITY|VIP|SHIP)_\d{3}\b")


def load_policy_documents(file_path: str) -> list[Document]:
    """按 ``## POLICY_001: 标题`` 切分 Markdown，并保留条款编号元数据。"""
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    documents: list[Document] = []
    document_preamble: list[str] = []
    for line in lines:
        if CLAUSE_HEADING.match(line):
            break
        document_preamble.append(line)
    current_id: str | None = None
    current_title = ""
    current_lines: list[str] = []

    document_rules = "\n".join(document_preamble).strip()
    # 文档级优先级规则单独入库：复制到每个条款会稀释条款本体的向量语义。
    # FAQ 前言只是编辑说明，不是可用于决策的业务规则，因此不建立该类 chunk。
    if document_rules and path.name != "06_faq.md":
        documents.append(Document(
            page_content=f"【文档级规则】\n{document_rules}",
            metadata={
                "source": path.name,
                "clause_ids": [],
                "clause_title": "文档级规则与优先级",
                "source_type": "policy_rule",
                "canonical_clause_ids": [],
                "document_rules": document_rules,
            },
        ))

    def flush() -> None:
        if current_id and (content := "\n".join(current_lines).strip()):
            canonical_clause_ids = []
            if path.name == "06_faq.md":
                canonical_clause_ids = sorted(set(CANONICAL_CLAUSE_ID.findall(content)))
            documents.append(Document(
                page_content=f"## {current_id}: {current_title}\n\n{content}",
                metadata={
                    "source": path.name,
                    "clause_ids": [current_id],
                    "clause_title": current_title,
                    "source_type": "faq" if path.name == "06_faq.md" else "policy",
                    "canonical_clause_ids": canonical_clause_ids,
                    "document_rules": document_rules,
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
