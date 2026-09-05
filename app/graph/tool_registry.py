"""Capability registry and deterministic guard for agent tools."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class ToolCode(str, Enum):
    SUCCESS = "SUCCESS"
    ELIGIBILITY_PASSED = "ELIGIBILITY_PASSED"
    ELIGIBILITY_REJECTED = "ELIGIBILITY_REJECTED"
    REFUND_SUBMITTED = "REFUND_SUBMITTED"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    MISSING_SLOT = "MISSING_SLOT"
    NOT_AUTHORIZED = "NOT_AUTHORIZED"
    INVALID_STAGE = "INVALID_STAGE"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    BUSINESS_REJECTED = "BUSINESS_REJECTED"
    TEMPORARY_FAILURE = "TEMPORARY_FAILURE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    code: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    retryable: bool = False
    idempotency_key: str | None = None


Handler = Callable[..., Awaitable[ToolOutcome]]


@dataclass(frozen=True)
class ToolCapability:
    name: str
    domain: str
    allowed_stages: frozenset[str]
    required_slots: frozenset[str] = frozenset()
    required_eligibility: str | None = None
    requires_user_confirmation: bool = False
    idempotency_key_template: str | None = None
    writes_to_conversation: bool = False
    audit_level: str = "read"
    owner_workflow: str = ""


class ToolCapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, ToolCapability] = {}
        self._handlers: dict[str, Handler] = {}

    def register(self, capability: ToolCapability, handler: Handler) -> None:
        if capability.name in self._capabilities:
            raise ValueError(f"duplicate tool capability: {capability.name}")
        if not callable(handler):
            raise TypeError(f"handler missing for {capability.name}")
        if not capability.allowed_stages:
            raise ValueError(f"allowed_stages missing for {capability.name}")
        if capability.idempotency_key_template and not capability.writes_to_conversation:
            raise ValueError("idempotency keys are only valid for write capabilities")
        self._capabilities[capability.name] = capability
        self._handlers[capability.name] = handler

    def get(self, name: str) -> ToolCapability:
        try:
            return self._capabilities[name]
        except KeyError as exc:
            raise ValueError(f"unregistered tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._capabilities)

    async def invoke(self, name: str, state: dict[str, Any], **arguments: Any) -> ToolOutcome:
        capability = self.get(name)
        domain = state.get("active_domain") or state.get("intent")
        if domain and domain != capability.domain:
            return ToolOutcome(False, ToolCode.INVALID_STAGE, "工具不属于当前业务领域")
        stage = state.get("workflow_stage")
        if stage not in capability.allowed_stages:
            return ToolOutcome(False, ToolCode.INVALID_STAGE, f"当前阶段不可调用 {name}")
        slots = state.get("collected_slots") if isinstance(state.get("collected_slots"), dict) else {}
        missing = [key for key in capability.required_slots if not state.get(key) and not slots.get(key)]
        if missing:
            return ToolOutcome(False, ToolCode.MISSING_SLOT, f"缺少必要信息：{', '.join(sorted(missing))}")
        if capability.required_eligibility:
            result = state.get("last_tool_result") or {}
            if result.get("eligibility_passed") is not True:
                return ToolOutcome(False, ToolCode.BUSINESS_REJECTED, "退款资格尚未通过")
        if capability.requires_user_confirmation and not state.get("user_confirmed"):
            return ToolOutcome(False, ToolCode.NOT_CONFIRMED, "尚未完成用户确认")
        key = None
        if capability.idempotency_key_template:
            key = capability.idempotency_key_template.format(
                user_id=state.get("user_id"), order_id=state.get("active_order_id")
            )
        outcome = await self._handlers[name](state=state, idempotency_key=key, **arguments)
        return outcome if outcome.idempotency_key else ToolOutcome(
            outcome.ok, outcome.code, outcome.message, outcome.data, outcome.retryable, key
        )


class GuardedToolExecutor:
    """Shared execution facade used by workflows and compatibility adapters."""

    def __init__(self, registry: ToolCapabilityRegistry | None = None) -> None:
        self.registry = registry or tool_registry

    async def invoke(self, tool_name: str, state: dict[str, Any], arguments: dict[str, Any] | None = None) -> ToolOutcome:
        try:
            return await self.registry.invoke(tool_name, state, **(arguments or {}))
        except Exception as exc:  # keep tool failures structured at the boundary
            return ToolOutcome(False, ToolCode.SYSTEM_ERROR, "工具执行失败，请稍后重试", retryable=True,
                               data={"error_type": type(exc).__name__})


tool_registry = ToolCapabilityRegistry()
