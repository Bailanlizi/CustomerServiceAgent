import json
import re
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.conversation import ConversationSession

MEMORY_KEYS = (
    "active_domain", "active_order_id", "active_order_sn", "conversation_goal",
    "collected_slots", "pending_slots", "workflow_stage", "last_tool_result",
    "next_action", "conversation_summary",
)
SUMMARY_TRIGGER_TURNS = 12
SUMMARY_RETAIN_TURNS = 3
ORDER_SN_PATTERN = re.compile(r"(?<![A-Z0-9])SN\d+\b", re.IGNORECASE)


class ConversationNotFoundError(Exception):
    pass


class ConversationForbiddenError(Exception):
    pass


class ConversationMismatchError(Exception):
    pass


class SlotExtraction(BaseModel):
    domain: Literal["ORDER", "POLICY", "REFUND", "OTHER"]
    explicit_new_goal: bool = False
    conversation_goal: str | None = None
    refund_reason: str | None = None
    user_constraints: list[str] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    confirmed_facts: list[str] = Field(default_factory=list)
    completed_actions: list[str] = Field(default_factory=list)
    pending_items: list[str] = Field(default_factory=list)
    user_constraints: list[str] = Field(default_factory=list)

    def render(self) -> str:
        sections = (
            ("已确认事实", self.confirmed_facts),
            ("已完成动作", self.completed_actions),
            ("未决事项", self.pending_items),
            ("用户约束", self.user_constraints),
        )
        return "\n".join(f"{title}：{'；'.join(items) or '无'}" for title, items in sections)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def default_working_memory() -> dict[str, Any]:
    return {
        "active_domain": None,
        "active_order_id": None,
        "active_order_sn": None,
        "conversation_goal": None,
        "collected_slots": {},
        "pending_slots": [],
        "workflow_stage": None,
        "last_tool_result": None,
        "next_action": None,
        "conversation_summary": None,
    }


class ConversationStateManager:
    """Creates/restores sessions and safely merges persistent working memory."""

    def __init__(self, extractor=None, summarizer=None):
        base_llm = ChatOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=SecretStr(settings.OPENAI_API_KEY),
            model=settings.LLM_MODEL,
            temperature=0,
        )
        self.extractor = extractor or base_llm.with_structured_output(SlotExtraction)
        self.summarizer = summarizer or base_llm.with_structured_output(ConversationSummary)

    async def resolve_session(
        self,
        user_id: int,
        client_session_id: str,
        conversation_id: UUID | None = None,
    ) -> ConversationSession:
        async with async_session_maker() as db:
            if conversation_id:
                item = await db.get(ConversationSession, conversation_id)
                if item is None:
                    raise ConversationNotFoundError
                if item.user_id != user_id:
                    raise ConversationForbiddenError
                if item.client_session_id != client_session_id:
                    raise ConversationMismatchError
                return item

            result = await db.exec(
                select(ConversationSession).where(
                    ConversationSession.user_id == user_id,
                    ConversationSession.client_session_id == client_session_id,
                )
            )
            existing = result.first()
            if existing:
                return existing

            generated_id = uuid4()
            item = ConversationSession(
                conversation_id=generated_id,
                user_id=user_id,
                client_session_id=client_session_id,
                checkpoint_thread_id=f"conversation:{generated_id}",
                working_memory_json=default_working_memory(),
            )
            db.add(item)
            try:
                await db.commit()
                await db.refresh(item)
                return item
            except IntegrityError:
                await db.rollback()
                result = await db.exec(
                    select(ConversationSession).where(
                        ConversationSession.user_id == user_id,
                        ConversationSession.client_session_id == client_session_id,
                    )
                )
                raced = result.first()
                if raced is None:
                    raise
                return raced

    async def prepare_turn(self, session: ConversationSession, question: str) -> dict[str, Any]:
        memory = default_working_memory()
        memory.update(session.working_memory_json or {})
        memory["conversation_summary"] = session.conversation_summary

        explicit_order = ORDER_SN_PATTERN.search(question)
        if explicit_order:
            order_sn = explicit_order.group(0).upper()
            if order_sn != memory.get("active_order_sn"):
                memory["active_order_id"] = None
            memory["active_order_sn"] = order_sn
            slots = dict(memory.get("collected_slots") or {})
            slots["order_sn"] = order_sn
            memory["collected_slots"] = slots

        extraction = await self._extract(question, memory)
        previous_domain = memory.get("active_domain")
        candidate = extraction.domain
        compatible_followup = previous_domain == "REFUND" and (
            bool(explicit_order) or bool(extraction.refund_reason) or candidate in {"REFUND", "OTHER"}
        )
        if previous_domain and (
            (compatible_followup and not extraction.explicit_new_goal) or candidate == "OTHER"
        ):
            candidate = previous_domain

        if candidate in {"ORDER", "POLICY", "REFUND"}:
            memory["active_domain"] = candidate
        if extraction.conversation_goal:
            memory["conversation_goal"] = extraction.conversation_goal

        slots = dict(memory.get("collected_slots") or {})
        if extraction.refund_reason:
            slots["refund_reason"] = extraction.refund_reason
        if extraction.user_constraints:
            slots["user_constraints"] = list(dict.fromkeys(
                [*slots.get("user_constraints", []), *extraction.user_constraints]
            ))
        memory["collected_slots"] = slots

        if memory["active_domain"] == "REFUND":
            pending = []
            if not memory.get("active_order_sn"):
                pending.append("order_sn")
            if not slots.get("refund_reason"):
                pending.append("refund_reason")
            memory["pending_slots"] = pending
            memory["next_action"] = "ASK" if pending else memory.get("next_action")

        return {
            **{key: memory.get(key) for key in MEMORY_KEYS},
            "intent": memory.get("active_domain"),
        }

    async def _extract(self, question: str, memory: dict[str, Any]) -> SlotExtraction:
        prompt = (
            "从用户最新一句话提取客服流程信息。domain 只能是 ORDER/POLICY/REFUND/OTHER；"
            "只有用户清楚提出不同任务时 explicit_new_goal 才为 true；退款原因必须是用户明确表达的原因，"
            "不得推测。\n当前记忆：" + json.dumps(memory, ensure_ascii=False, default=str)
            + "\n最新输入：" + question
        )
        try:
            raw = await self.extractor.ainvoke([HumanMessage(content=prompt)])
            return SlotExtraction.model_validate(raw)
        except Exception:  # noqa: BLE001 - deterministic extraction is the availability fallback.
            # Continuity must remain available when the extraction model is unavailable.
            if any(word in question for word in ("退款", "退货", "换货")):
                domain = "REFUND"
            elif any(word in question for word in ("订单", "物流", "快递")):
                domain = "ORDER"
            elif any(word in question for word in ("政策", "规定", "运费", "多久")):
                domain = "POLICY"
            else:
                domain = "OTHER"
            return SlotExtraction(domain=domain, explicit_new_goal=False)

    async def persist_turn(self, session: ConversationSession, state: dict[str, Any]) -> bool:
        memory = default_working_memory()
        memory.update(session.working_memory_json or {})
        for key in MEMORY_KEYS:
            if key in state:
                memory[key] = state[key]
        order_data = state.get("order_data")
        if isinstance(order_data, dict) and order_data.get("id") and order_data.get("order_sn"):
            memory["active_order_id"] = order_data["id"]
            memory["active_order_sn"] = order_data["order_sn"]
            memory["collected_slots"] = {
                **(memory.get("collected_slots") or {}), "order_sn": order_data["order_sn"]
            }

        messages = list(state.get("messages") or [])
        summary = session.conversation_summary
        compacted = False
        if self._human_turn_count(messages) >= SUMMARY_TRIGGER_TURNS:
            generated = await self._summarize(messages, memory, summary)
            if generated:
                summary = generated
                memory["conversation_summary"] = generated
                compacted = True

        async with async_session_maker() as db:
            result = await db.exec(
                select(ConversationSession)
                .where(ConversationSession.conversation_id == session.conversation_id)
                .with_for_update()
            )
            current = result.one()
            latest = default_working_memory()
            latest.update(current.working_memory_json or {})
            if current.version == session.version:
                latest.update(memory)
            else:
                # A concurrent turn committed after this request started. Merge
                # additive facts and never replace its non-null results with a
                # stale null from our snapshot.
                for key, value in memory.items():
                    if value is not None:
                        latest[key] = value
            latest["collected_slots"] = {
                **((current.working_memory_json or {}).get("collected_slots") or {}),
                **(memory.get("collected_slots") or {}),
            }
            current.working_memory_json = dict(latest)
            current.active_domain = latest.get("active_domain")
            current.conversation_summary = summary
            current.last_active_at = _utcnow()
            current.updated_at = _utcnow()
            current.version += 1
            db.add(current)
            await db.commit()
        return compacted

    async def _summarize(
        self, messages: list[BaseMessage], memory: dict[str, Any], previous: str | None
    ) -> str | None:
        prompt = (
            "把对话压缩为四类结构化信息。必须原样保留业务ID、未决槽位、工具失败原因和用户明确约束。\n"
            f"旧摘要：{previous or '无'}\n工作记忆：{json.dumps(memory, ensure_ascii=False, default=str)}\n"
            "对话：\n" + "\n".join(f"{m.type}: {m.content}" for m in messages)
        )
        try:
            raw = await self.summarizer.ainvoke([HumanMessage(content=prompt)])
            result = ConversationSummary.model_validate(raw)
            rendered = result.render()
            required = [
                str(value) for value in (
                    memory.get("active_order_id"), memory.get("active_order_sn")
                ) if value is not None
            ]
            if any(value not in rendered for value in required):
                return None
            return rendered
        except Exception:  # noqa: BLE001 - failed compaction must never lose conversation state.
            return None

    @staticmethod
    def _human_turn_count(messages: list[BaseMessage]) -> int:
        return sum(isinstance(message, HumanMessage) for message in messages)

    @staticmethod
    def retained_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
        human_indexes = [i for i, item in enumerate(messages) if isinstance(item, HumanMessage)]
        if len(human_indexes) <= SUMMARY_RETAIN_TURNS:
            return messages
        return messages[human_indexes[-SUMMARY_RETAIN_TURNS]:]
