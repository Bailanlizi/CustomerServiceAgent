import pytest

from app.graph import nodes
from app.services.policy_answer_guard import (
    NO_EVIDENCE_FALLBACK,
    SAFE_POLICY_FALLBACK,
    PolicyAnswer,
    PolicyCitationValidationError,
    allowed_evidence_ids,
    validate_policy_answer,
)

EVIDENCE = [
    {
        "content": "FAQ 说明",
        "source": "06_faq.md",
        "clause_ids": ["FAQ_016"],
        "canonical_clause_ids": ["QUALITY_004"],
        "source_type": "faq",
        "rank": 1,
        "distance": 0.1,
    },
    {
        "content": "正式政策",
        "source": "03_quality_return_policy.md",
        "clause_ids": ["QUALITY_004"],
        "canonical_clause_ids": [],
        "source_type": "policy",
        "rank": 2,
        "distance": 0.2,
    },
]


def test_allowed_evidence_includes_retrieved_and_faq_canonical_ids():
    assert allowed_evidence_ids(EVIDENCE) == ["FAQ_016", "QUALITY_004"]


def test_policy_answer_accepts_direct_and_canonical_evidence_ids():
    verified = validate_policy_answer(
        PolicyAnswer(
            answer="质量问题可在签收后 30 天内申请。",
            applied_clause_ids=["QUALITY_004"],
            evidence_clause_ids=["FAQ_016", "QUALITY_004"],
        ),
        allowed_evidence_ids(EVIDENCE),
        evidence_present=True,
    )

    assert verified.applied_clause_ids == ["QUALITY_004"]


@pytest.mark.parametrize(
    "candidate",
    [
        PolicyAnswer(answer="可以处理。", applied_clause_ids=["RETURN_999"], evidence_clause_ids=["RETURN_999"]),
        PolicyAnswer(answer="可以处理。", applied_clause_ids=["QUALITY_004"], evidence_clause_ids=["FAQ_016"]),
        PolicyAnswer(answer="可以处理。", applied_clause_ids=[], evidence_clause_ids=[]),
        PolicyAnswer(answer="依据 RETURN_999 可以处理。", applied_clause_ids=["QUALITY_004"], evidence_clause_ids=["QUALITY_004"]),
    ],
)
def test_policy_answer_rejects_invalid_or_missing_citations(candidate):
    with pytest.raises(PolicyCitationValidationError):
        validate_policy_answer(candidate, allowed_evidence_ids(EVIDENCE), evidence_present=True)


class StubPolicyAnswerLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.seen_messages: list[list] = []

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        self.seen_messages.append(list(messages))
        return next(self.responses)


def _state():
    return {
        "question": "质量问题多久可以退？",
        "policy_evidence": EVIDENCE,
        "policy_rules": [],
    }


@pytest.mark.asyncio
async def test_policy_generation_records_passed_audit(monkeypatch):
    stub = StubPolicyAnswerLLM([
        PolicyAnswer(
            answer="质量问题可在签收后 30 天内申请。",
            applied_clause_ids=["QUALITY_004"],
            evidence_clause_ids=["QUALITY_004"],
        )
    ])
    monkeypatch.setattr(nodes, "policy_answer_llm", stub)

    result = await nodes._generate_verified_policy_answer(_state())

    assert result["answer"].startswith("质量问题")
    assert result["policy_answer_audit"]["status"] == "passed"
    assert result["policy_answer_audit"]["retry_count"] == 0


@pytest.mark.asyncio
async def test_policy_generation_retries_invalid_citation(monkeypatch):
    stub = StubPolicyAnswerLLM([
        PolicyAnswer(answer="可以退。", applied_clause_ids=["RETURN_999"], evidence_clause_ids=["RETURN_999"]),
        PolicyAnswer(answer="质量问题可在签收后 30 天内申请。", applied_clause_ids=["QUALITY_004"], evidence_clause_ids=["QUALITY_004"]),
    ])
    monkeypatch.setattr(nodes, "policy_answer_llm", stub)

    result = await nodes._generate_verified_policy_answer(_state())

    assert stub.calls == 2
    assert result["policy_answer_audit"]["status"] == "passed"
    assert result["policy_answer_audit"]["retry_count"] == 1


@pytest.mark.asyncio
async def test_policy_generation_uses_safe_fallback_after_two_invalid_attempts(monkeypatch):
    stub = StubPolicyAnswerLLM([
        PolicyAnswer(answer="可以退。", applied_clause_ids=["RETURN_999"], evidence_clause_ids=["RETURN_999"]),
        PolicyAnswer(answer="依据 RETURN_999 可以退。", applied_clause_ids=["QUALITY_004"], evidence_clause_ids=["QUALITY_004"]),
    ])
    monkeypatch.setattr(nodes, "policy_answer_llm", stub)

    result = await nodes._generate_verified_policy_answer(_state())

    assert result["answer"] == SAFE_POLICY_FALLBACK
    assert result["policy_answer_audit"]["status"] == "fallback"


def test_policy_answer_rejects_citations_without_evidence():
    """无检索证据时引用任何条款都视为编造，必须被拒绝。"""
    with pytest.raises(PolicyCitationValidationError):
        validate_policy_answer(
            PolicyAnswer(
                answer="可以的，30 天内都能退。",
                applied_clause_ids=["RETURN_001"],
                evidence_clause_ids=["RETURN_001"],
            ),
            allowed_ids=[],
            evidence_present=False,
        )


@pytest.mark.asyncio
async def test_no_evidence_short_circuits_to_safe_answer(monkeypatch):
    """检索不到证据时不调用 LLM，直接返回确定性安全答复（堵死过度承诺路径）。"""
    stub = StubPolicyAnswerLLM([])
    monkeypatch.setattr(nodes, "policy_answer_llm", stub)

    state = {**_state(), "policy_evidence": []}
    result = await nodes._generate_verified_policy_answer(state)

    assert stub.calls == 0
    assert result["answer"] == NO_EVIDENCE_FALLBACK
    assert result["policy_answer_audit"]["status"] == "no_evidence"
    assert result["policy_answer_audit"]["retry_count"] == 0


@pytest.mark.asyncio
async def test_retry_feeds_validation_error_back_to_model(monkeypatch):
    """第二次调用必须携带第一次的校验错误，让模型知道该改什么。"""
    stub = StubPolicyAnswerLLM([
        PolicyAnswer(answer="可以退。", applied_clause_ids=["RETURN_999"], evidence_clause_ids=["RETURN_999"]),
        PolicyAnswer(answer="质量问题可在签收后 30 天内申请。", applied_clause_ids=["QUALITY_004"], evidence_clause_ids=["QUALITY_004"]),
    ])
    monkeypatch.setattr(nodes, "policy_answer_llm", stub)

    result = await nodes._generate_verified_policy_answer(_state())

    assert stub.calls == 2
    assert result["policy_answer_audit"]["status"] == "passed"
    # 第二次调用在原消息基础上追加了错误反馈消息。
    assert len(stub.seen_messages[1]) == len(stub.seen_messages[0]) + 1
    feedback = stub.seen_messages[1][-1]
    assert "未通过引用校验" in feedback.content
    assert "RETURN_999" in feedback.content
