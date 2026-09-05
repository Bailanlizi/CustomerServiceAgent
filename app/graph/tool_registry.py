"""Capability registry and deterministic guard for agent tools.

P2 design:

  * ToolCapabilityRegistry is the single entry point for *all* tool execution,
    whether the call originates from a LangChain ToolNode or from a workflow
    node. ToolNode wrappers in `app.graph.tools` simply forward to
    `GuardedToolExecutor.invoke`, so there is no parallel, unguarded path.
  * Handlers return a *partial* ToolOutcome (business result only). The
    registry fills in audit metadata (`tool_name`, `workflow_stage`,
    `conversation_id`, `thread_id`, `user_id`, `order_id`, `timestamp`,
    `domain`) before returning, so the FSM, audit log and admin queue all read
    from a single structured source instead of parsing Chinese strings.
  * Audit hooks: handlers whose `audit_level` is `sensitive` get a
    `decision_metadata` AuditLog row written on every outcome, including
    `BUSINESS_REJECTED` / `TEMPORARY_FAILURE` / `NOT_CONFIRMED` / `MISSING_SLOT`.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.database import async_session_maker
from app.models.audit import AuditAction, AuditLog


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
    """Structured tool result. Handlers return this; FSM / audit / admin read this.

    Fields are split into two groups:
      - *Business*: `ok`, `code`, `message`, `data`, `retryable`, `idempotency_key`.
        These are produced by the underlying service / tool handler.
      - *Audit metadata*: `tool_name`, `domain`, `workflow_stage`,
        `conversation_id`, `thread_id`, `user_id`, `order_id`, `timestamp`.
        These are filled in by `ToolCapabilityRegistry.invoke` so handlers
        don't have to plumb them through manually.
    """

    ok: bool
    code: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    retryable: bool = False
    idempotency_key: str | None = None
    # Audit metadata — populated by the registry, not by handlers.
    tool_name: str | None = None
    domain: str | None = None
    workflow_stage: str | None = None
    conversation_id: str | None = None
    thread_id: str | None = None
    user_id: int | None = None
    order_id: int | None = None
    timestamp: str | None = None


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
    audit_level: str = "read"  # "read" | "sensitive"
    owner_workflow: str = ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


class ToolCapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, ToolCapability] = {}
        self._handlers: dict[str, Handler] = {}
        # P1-3 收尾：进程内幂等命中缓存。
        # Registry 在 invoke 阶段为带 idempotency_key_template 的写工具生成
        # 键；同一 key 第二次调用时直接复用最近的 ToolOutcome，避免重复
        # 触发底层业务逻辑（退款提交、订单创建等）。这是与"业务层查询
        # RefundApplication"并行的第二道防线——Registry 层命中先于业务层。
        self._idempotency_cache: dict[str, ToolOutcome] = {}

    # ----------------------------------------------------------
    # Registration
    # ----------------------------------------------------------
    def register(self, capability: ToolCapability, handler: Handler) -> None:
        if capability.name in self._capabilities:
            raise ValueError(f"duplicate tool capability: {capability.name}")
        if not callable(handler):
            raise TypeError(f"handler missing for {capability.name}")
        if not capability.allowed_stages:
            raise ValueError(f"allowed_stages missing for {capability.name}")
        if capability.idempotency_key_template and not capability.writes_to_conversation:
            raise ValueError("idempotency keys are only valid for write capabilities")
        if capability.audit_level == "sensitive" and not capability.writes_to_conversation:
            raise ValueError("sensitive audit requires writes_to_conversation=True")
        self._capabilities[capability.name] = capability
        self._handlers[capability.name] = handler

    def get(self, name: str) -> ToolCapability:
        try:
            return self._capabilities[name]
        except KeyError as exc:
            raise ValueError(f"unregistered tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._capabilities)

    def capabilities(self) -> Iterable[ToolCapability]:
        return tuple(self._capabilities.values())

    # ----------------------------------------------------------
    # Execution
    # ----------------------------------------------------------
    async def invoke(
        self,
        name: str,
        state: dict[str, Any],
        **arguments: Any,
    ) -> ToolOutcome:
        capability = self.get(name)

        # 1. Domain check (allow tools whose domain matches active_domain or
        # intent, or unset).
        domain = state.get("active_domain") or state.get("intent")
        if domain and domain != capability.domain:
            return self._envelope(
                name,
                capability,
                state,
                ToolOutcome(
                    ok=False,
                    code=ToolCode.INVALID_STAGE,
                    message=f"工具不属于当前业务领域（{capability.domain}）",
                ),
            )

        # 2. Stage check.
        # `"*"` in allowed_stages is a wildcard — the tool is callable from any
        # workflow stage (used by read-only / domain-cross tools like
        # `query_order_tool`, which must be available regardless of where the
        # user is in the refund FSM).
        stage = state.get("workflow_stage")
        allowed = capability.allowed_stages
        if "*" not in allowed and stage not in allowed:
            return self._envelope(
                name,
                capability,
                state,
                ToolOutcome(
                    ok=False,
                    code=ToolCode.INVALID_STAGE,
                    message=f"当前阶段不可调用 {name}",
                ),
            )

        # 3. Required slots.
        slots = state.get("collected_slots") if isinstance(state.get("collected_slots"), dict) else {}
        missing = [
            key for key in capability.required_slots
            if not state.get(key) and not slots.get(key)
        ]
        if missing:
            return self._envelope(
                name,
                capability,
                state,
                ToolOutcome(
                    ok=False,
                    code=ToolCode.MISSING_SLOT,
                    message=f"缺少必要信息：{', '.join(sorted(missing))}",
                ),
            )

        # 4. Eligibility check.
        if capability.required_eligibility:
            last_result = state.get("last_tool_result")
            eligibility_passed = (
                isinstance(last_result, dict)
                and last_result.get("eligibility_passed") is True
            )
            if not eligibility_passed:
                return self._envelope(
                    name,
                    capability,
                    state,
                    ToolOutcome(
                        ok=False,
                        code=ToolCode.BUSINESS_REJECTED,
                        message="退款资格尚未通过",
                    ),
                )

        # 5. User confirmation.
        if capability.requires_user_confirmation and not state.get("user_confirmed"):
            return self._envelope(
                name,
                capability,
                state,
                ToolOutcome(
                    ok=False,
                    code=ToolCode.NOT_CONFIRMED,
                    message="尚未完成用户确认",
                ),
            )

        # 6. Compute idempotency key (only valid for write tools; validated at register time).
        key = None
        if capability.idempotency_key_template:
            key = capability.idempotency_key_template.format(
                user_id=state.get("user_id"),
                order_id=state.get("active_order_id"),
            )
            # 6a. Idempotency hit: 同一 key 在本次进程内已有最近成功 outcome
            # 时直接返回，避免重复触发业务逻辑（与下层"查询
            # RefundApplication"防线并行）。
            cached = self._idempotency_cache.get(key)
            if cached is not None and cached.ok:
                return self._envelope(
                    name,
                    capability,
                    state,
                    replace(cached, idempotency_key=key, timestamp=_utc_now_iso()),
                    idempotency_key=key,
                )

        # 7. Execute handler.
        try:
            raw = await self._handlers[name](state=state, idempotency_key=key, **arguments)
        except Exception as exc:  # noqa: BLE001 - registry boundary.
            return self._envelope(
                name,
                capability,
                state,
                ToolOutcome(
                    ok=False,
                    code=ToolCode.SYSTEM_ERROR,
                    message=f"工具执行失败：{exc}",
                    retryable=True,
                    data={"error_type": type(exc).__name__},
                ),
                idempotency_key=key,
            )

        # 8. Handler returns a partial outcome; envelope it with metadata + idempotency key.
        enveloped = self._envelope(
            name,
            capability,
            state,
            raw,
            idempotency_key=key,
        )
        # 8a. Idempotency 缓存：仅记录成功 outcome，确保后续重放命中。
        if key is not None and enveloped.ok:
            self._idempotency_cache[key] = enveloped
        return enveloped

    # ----------------------------------------------------------
    # Internals
    # ----------------------------------------------------------
    def _envelope(
        self,
        name: str,
        capability: ToolCapability,
        state: dict[str, Any],
        outcome: ToolOutcome,
        idempotency_key: str | None = None,
    ) -> ToolOutcome:
        key = outcome.idempotency_key or idempotency_key
        return replace(
            outcome,
            idempotency_key=key,
            tool_name=name,
            domain=capability.domain,
            workflow_stage=state.get("workflow_stage"),
            conversation_id=state.get("conversation_id"),
            thread_id=state.get("thread_id"),
            user_id=state.get("user_id"),
            order_id=state.get("active_order_id"),
            timestamp=_utc_now_iso(),
        )


class GuardedToolExecutor:
    """Shared execution facade used by workflows and LangChain compatibility wrappers.

    All paths (workflow nodes + ToolNode) must funnel through here so that
    `ToolCapabilityRegistry.invoke` is the single arbiter of "can this tool be
    called now, by this user, with these slots?".
    """

    def __init__(self, registry: ToolCapabilityRegistry | None = None) -> None:
        self.registry = registry or tool_registry

    async def invoke(
        self,
        tool_name: str,
        state: dict[str, Any],
        arguments: dict[str, Any] | None = None,
    ) -> ToolOutcome:
        try:
            return await self.registry.invoke(tool_name, state, **(arguments or {}))
        except Exception as exc:  # noqa: BLE001 - last-line defence.
            return ToolOutcome(
                ok=False,
                code=ToolCode.SYSTEM_ERROR,
                message="工具执行失败，请稍后重试",
                retryable=True,
                data={"error_type": type(exc).__name__},
            )


# ==========================================================
# Audit hook (used by call sites when audit_level=sensitive).
# ==========================================================
async def write_tool_audit_log(
    outcome: ToolOutcome,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> int | None:
    """Persist an AuditLog row capturing the tool outcome.

    Called for every sensitive audit tool outcome (success, rejection, retry).
    `decision_metadata` and `context_snapshot` are filled with the structured
    fields so that the existing audit UI can render them without parsing strings.

    Returns the AuditLog.id when written, or None if persistence failed
    (audit failures must not roll back the business outcome).
    """
    metadata = {
        "tool_name": outcome.tool_name,
        "domain": outcome.domain,
        "workflow_stage": outcome.workflow_stage,
        "outcome_code": outcome.code,
        "ok": outcome.ok,
        "retryable": outcome.retryable,
        "idempotency_key": outcome.idempotency_key,
        "timestamp": outcome.timestamp,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    snapshot = {
        "tool_name": outcome.tool_name,
        "domain": outcome.domain,
        "workflow_stage": outcome.workflow_stage,
        "outcome_code": outcome.code,
        "ok": outcome.ok,
        "message": outcome.message,
        "data": outcome.data,
        "retryable": outcome.retryable,
        "idempotency_key": outcome.idempotency_key,
        "timestamp": outcome.timestamp,
        "user_id": outcome.user_id,
        "order_id": outcome.order_id,
    }

    try:
        async with async_session_maker() as session:  # type: AsyncSession
            audit_log = AuditLog(
                thread_id=outcome.thread_id,
                user_id=outcome.user_id,
                order_id=outcome.order_id,
                refund_application_id=(outcome.data or {}).get("refund_id"),
                trigger_reason=f"{outcome.tool_name} -> {outcome.code}",
                risk_level="LOW",
                action=AuditAction.PENDING if outcome.ok else AuditAction.REJECT,
                decision_metadata=metadata,
                context_snapshot=snapshot,
            )
            session.add(audit_log)
            await session.commit()
            await session.refresh(audit_log)
            return int(audit_log.id)
    except Exception as exc:  # noqa: BLE001 - audit failures must not roll back business outcome.
        print(f"[tool_registry] write_tool_audit_log failed: {exc}")
        return None


tool_registry = ToolCapabilityRegistry()


# Compatibility: keep the module-level reference reachable as before.
__all__ = [
    "GuardedToolExecutor",
    "Handler",
    "ToolCapability",
    "ToolCapabilityRegistry",
    "ToolCode",
    "ToolOutcome",
    "tool_registry",
    "write_tool_audit_log",
]