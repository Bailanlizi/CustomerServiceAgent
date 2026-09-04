"""政策回答的证据引用约束与确定性校验。"""

import re
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, Field

# 使用通用的大写条款编号格式，避免模型在用户可见回答中编造新的政策类别。
CLAUSE_ID_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]*_\d{3}\b")
SAFE_POLICY_FALLBACK = "抱歉，当前无法依据已核验的政策信息作出确定承诺，请联系人工客服核实。"
NO_EVIDENCE_FALLBACK = "抱歉，暂未查询到与您问题相关的政策规定，无法作出确定答复。如需进一步确认，请联系人工客服。"

# 政策结构化生成调用的标签：chat 端据此丢弃该 LLM 的任何中间流式 token，
# 确保未通过校验的结构化输出（如 JSON 片段）不会泄露给前端。
POLICY_GUARD_TAG = "policy_guard"

# 内部结构化 LLM 调用的通用标签（意图识别、政策生成等）。
# chat 端据此丢弃这些调用的流式 token——它们只是内部决策/结构化输出，
# 不是面向用户的答案，泄露会污染前端，且会误置 token_sent 导致真正的答案被吞掉。
INTERNAL_LLM_TAG = "internal_llm"


class PolicyAnswer(BaseModel):
    """模型生成的、待后端校验的政策回答。"""

    answer: str = Field(description="面向用户的自然语言回答，不展示条款编号。")
    applied_clause_ids: list[str] = Field(
        default_factory=list,
        description="直接用于得出结论的条款编号。",
    )
    evidence_clause_ids: list[str] = Field(
        default_factory=list,
        description="支撑解释的全部条款编号。",
    )


class PolicyCitationValidationError(ValueError):
    """政策回答引用不满足可审计约束。"""


def allowed_evidence_ids(evidence: Iterable[Mapping[str, object]]) -> list[str]:
    """从检索证据及 FAQ 的 canonical 映射构建稳定的合法引用集合。"""
    allowed: set[str] = set()
    for item in evidence:
        for field in ("clause_ids", "canonical_clause_ids"):
            values = item.get(field, [])
            if isinstance(values, list):
                allowed.update(str(value) for value in values)
    return sorted(allowed)


def _normalise_ids(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(clause_id.strip() for clause_id in ids if clause_id.strip()))


def validate_policy_answer(
    candidate: PolicyAnswer,
    allowed_ids: Iterable[str],
    *,
    evidence_present: bool,
) -> PolicyAnswer:
    """验证结构化引用和用户可见回答中的所有条款编号。"""
    if not candidate.answer.strip():
        raise PolicyCitationValidationError("回答不能为空")

    allowed = set(allowed_ids)
    applied = _normalise_ids(candidate.applied_clause_ids)
    evidence = _normalise_ids(candidate.evidence_clause_ids)
    answer_ids = set(CLAUSE_ID_PATTERN.findall(candidate.answer))

    if not evidence_present:
        # 无检索证据时引用任何条款都视为编造，直接拒绝。
        if applied or evidence:
            raise PolicyCitationValidationError(
                f"没有检索证据时不得引用任何条款，却引用了: {sorted(set(applied) | set(evidence))}"
            )
    elif not applied or not evidence:
        raise PolicyCitationValidationError("存在检索证据时必须给出实际适用条款和证据条款")
    if not set(applied).issubset(evidence):
        raise PolicyCitationValidationError(
            f"实际适用条款必须是证据条款的子集，越界编号: {sorted(set(applied) - set(evidence))}"
        )
    if not set(evidence).issubset(allowed):
        raise PolicyCitationValidationError(
            f"证据条款包含未检索到的编号: {sorted(set(evidence) - allowed)}"
        )
    if answer_ids:
        raise PolicyCitationValidationError(
            f"用户可见回答不得展示条款编号: {sorted(answer_ids)}"
        )

    return PolicyAnswer(
        answer=candidate.answer.strip(),
        applied_clause_ids=applied,
        evidence_clause_ids=evidence,
    )
