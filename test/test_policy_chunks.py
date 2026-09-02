from pathlib import Path

from app.services.policy_chunks import load_policy_documents


ROOT = Path(__file__).resolve().parents[1]


def test_document_priority_rules_are_indexed_without_polluting_clause_chunks() -> None:
    chunks = load_policy_documents(str(ROOT / "data" / "04_vip_service_policy.md"))
    vip_003 = next(chunk for chunk in chunks if chunk.metadata["clause_ids"] == ["VIP_003"])
    priority_rule = next(chunk for chunk in chunks if chunk.metadata["source_type"] == "policy_rule")

    assert "优先级高于《通用退换货政策》" in priority_rule.page_content
    assert "退换货免运费" in vip_003.page_content
    assert vip_003.metadata["source_type"] == "policy"


def test_faq_chunks_record_explicit_canonical_policy_references() -> None:
    chunks = load_policy_documents(str(ROOT / "data" / "06_faq.md"))
    faq_016 = next(chunk for chunk in chunks if chunk.metadata["clause_ids"] == ["FAQ_016"])
    faq_008 = next(chunk for chunk in chunks if chunk.metadata["clause_ids"] == ["FAQ_008"])

    assert faq_016.metadata["source_type"] == "faq"
    assert faq_016.metadata["canonical_clause_ids"] == ["QUALITY_004"]
    assert faq_008.metadata["canonical_clause_ids"] == []
