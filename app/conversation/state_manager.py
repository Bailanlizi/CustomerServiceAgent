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
from app.models.conversation import ConversationMessage, ConversationSession
from app.models.order import Order

MEMORY_KEYS = (
    "active_domain", "active_order_id", "active_order_sn", "conversation_goal",
    "collected_slots", "pending_slots", "workflow_stage", "last_tool_result",
    "next_action", "conversation_summary", "refund_reason_category",
    # P1: 顶层 user_confirmed 与 collected_slots.user_confirmed 互相同步；
    # 顶层用于工具签名 InjectedState，collected_slots 是用户确认意图的真源。
    "user_confirmed",
)
SUMMARY_TRIGGER_TURNS = 6
# P1 修复: 客服平均对话 4-8 轮，12 触发偏晚；保留 3 轮又可能裁掉 ORDER 查询
# 中间事实（例如 6 轮里用户先查 SN001 又换 SN002，仅靠摘要字段难以定位）。
# 阈值降到 6、保留 5：保证用户在平均一轮对话长度结束前触发压缩；
# 同时保留足够多的最近消息让模型可以原样读到上一轮的订单号与问题。
SUMMARY_RETAIN_TURNS = 5
# 摘要置顶前缀：在 _summarize 渲染时强制拼到第一行，避免 LLM 把活跃订单号
# 压缩掉后导致下一轮模型忘记当前 case 的 SN。
SUMMARY_PINNED_SN_PREFIX = "【置顶】当前活跃订单号："
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
    # P1 修复: 退款原因分类由 LLM 一次产出，避免后续在工具侧再做 free-text → enum
    # 映射时漏匹配。值集与 RefundReason 枚举严格对齐，模型只能在列出的字面量中选择。
    refund_reason_category: Literal[
        "QUALITY_ISSUE", "SIZE_NOT_FIT", "NOT_AS_DESCRIBED",
        "CHANGED_MIND", "OTHER",
    ] | None = None
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
        "refund_reason_category": None,
        # P1: 顶层 user_confirmed 是工具签名 InjectedState("user_confirmed") 的入口；
        # 真源仍在 collected_slots.user_confirmed，由 prepare_turn / persist_turn 双向同步。
        "user_confirmed": None,
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
                return item

            result = await db.exec(
                select(ConversationSession).where(
                    ConversationSession.user_id == user_id,
                ).order_by(
                    ConversationSession.last_active_at.desc(),
                    ConversationSession.updated_at.desc(),
                ).limit(1)
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
                # A different order starts a new refund case; do not inherit
                # terminal, confirmation, reason, or eligibility state.
                slots = dict(memory.get("collected_slots") or {})
                for key in ("refund_submitted_id", "user_confirmed", "refund_reason", "order_sn"):
                    slots.pop(key, None)
                memory["collected_slots"] = slots
                memory["user_confirmed"] = None
                memory["refund_reason_category"] = None
                memory["last_tool_result"] = None
            memory["active_order_sn"] = order_sn
            slots = dict(memory.get("collected_slots") or {})
            slots["order_sn"] = order_sn
            memory["collected_slots"] = slots

        extraction = await self._extract(question, memory)
        previous_domain = memory.get("active_domain")
        candidate = extraction.domain
        # P1 修复: 用户明确表达目标切换（"我要退款/查订单/问政策"）时，绝不能被
        # 兜底到 previous_domain。只有当 LLM 给出明确的领域分类时，candidate 才允许
        # 无脑覆盖之前的活跃领域；OTHER 仍保持兜底，不强行切换避免误判。
        explicit_new_destination = (
            extraction.explicit_new_goal and candidate in {"ORDER", "POLICY", "REFUND"}
        )
        if not explicit_new_destination:
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
        # P1: 收集阶段已经把 collected_slots.user_confirmed 写好；这里把它提升到
        # 顶层 user_confirmed，让 submit_refund_application 通过 InjectedState 拿到。
        # 真源仍是 collected_slots，persist_turn 时再降级回去。
        if slots.get("user_confirmed"):
            memory["user_confirmed"] = True
        memory["collected_slots"] = slots
        # P1 修复: 将退款原因分类写入 working memory，让 submit_refund_application
        # 通过 InjectedState 直接消费，避免 LLM 在工具调用时再次做 free-text → enum 映射。
        if extraction.refund_reason_category:
            memory["refund_reason_category"] = extraction.refund_reason_category

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

    async def get_messages(self, conversation_id: UUID, user_id: int, limit: int = 100) -> list[ConversationMessage]:
        async with async_session_maker() as db:
            result = await db.exec(
                select(ConversationMessage)
                .where(
                    ConversationMessage.conversation_id == conversation_id,
                    ConversationMessage.user_id == user_id,
                )
                .order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
                .limit(limit)
            )
            return list(reversed(result.all()))

    async def append_messages(
        self, conversation_id: UUID, user_id: int, user_content: str, assistant_content: str
    ) -> None:
        async with async_session_maker() as db:
            db.add_all([
                ConversationMessage(conversation_id=conversation_id, user_id=user_id, role="user", content=user_content),
                ConversationMessage(conversation_id=conversation_id, user_id=user_id, role="assistant", content=assistant_content),
            ])
            await db.commit()

    async def _extract(self, question: str, memory: dict[str, Any]) -> SlotExtraction:
        prompt = (
            "从用户最新一句话提取客服流程信息。domain 只能是 ORDER/POLICY/REFUND/OTHER；"
            "只有用户清楚提出不同任务（如'我要退款''换个问题''我先问下运费'）时 explicit_new_goal 才为 true；"
            "退款原因必须是用户明确表达的原因，不得推测。"
            "若用户在退款语境下表达退款原因，必须同时给出 refund_reason_category，"
            "可选值严格为 QUALITY_ISSUE(质量问题) / SIZE_NOT_FIT(尺码不合适) / "
            "NOT_AS_DESCRIBED(与描述不符) / CHANGED_MIND(不想要了) / OTHER(其他)，"
            "不得使用其他字面量；判断不了分类时填 OTHER。\n"
            "当前记忆：" + json.dumps(memory, ensure_ascii=False, default=str)
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
        # P1: 顶层 user_confirmed 由子图/工具写入；这里降级到 collected_slots
        # 作为持久化真源。下一次 prepare_turn 再提升到顶层，避免工作记忆真源丢失。
        if memory.get("user_confirmed"):
            slots = dict(memory.get("collected_slots") or {})
            slots["user_confirmed"] = True
            memory["collected_slots"] = slots
        order_data = state.get("order_data")
        if isinstance(order_data, dict) and order_data.get("id") and order_data.get("order_sn"):
            memory["active_order_id"] = order_data["id"]
            memory["active_order_sn"] = order_data["order_sn"]
            memory["collected_slots"] = {
                **(memory.get("collected_slots") or {}), "order_sn": order_data["order_sn"]
            }
        # P1 修复: REFUND 路径下 state["order_data"] 永远是 None，active_order_id
        # 无法被持久化，导致 admin 工作台按 case 维度聚合时缺关键字段。
        # 这里在回合结束前基于 active_order_sn + user_id 反查一次 Order，得到的
        # id 写回 working memory。下一次 checkout / submit 时工具无需再让 LLM
        # 重新声明订单号，避免"工作记忆里有了订单号但工具还要再问"的断点。
        if memory.get("active_order_sn") and not memory.get("active_order_id"):
            user_id = state.get("user_id")
            if user_id is not None:
                order_id = await self._resolve_order_id(
                    memory["active_order_sn"], user_id
                )
                if order_id is not None:
                    memory["active_order_id"] = order_id

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

    @staticmethod
    async def _resolve_order_id(order_sn: str, user_id: int) -> int | None:
        """根据订单号 + 用户 ID 反查订单主键。

        仅在 active_order_id 缺失时调用，目的不是替代实时查询，而是把"工作记忆"
        里已经确认的订单号固化成一个稳定的内部主键，供 P1 工具注册表校验、
        审计日志关联和 admin 工作台按 case 维度聚合使用。查不到时返回 None，
        不抛异常——下次回合由 REFUND 工具再次校验即可。
        """
        async with async_session_maker() as db:
            stmt = select(Order).where(
                Order.order_sn == order_sn,
                Order.user_id == user_id,
            )
            result = await db.exec(stmt)
            order = result.first()
        return order.id if order else None

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
            # P1 修复: 强制把 active_order_sn 置顶在摘要第一行，避免长对话
            # 压缩后下一轮模型忘记当前 case 的 SN。这里直接拼接，不依赖 LLM
            # 输出：1) SN 永远在最前不会被其他事实抢位；2) 即便保留的消息窗口
            # 已经裁掉原 SN 消息，摘要里仍能找回；3) required 校验仍可放行，
            # 因为我们主动把 SN 写进了 rendered。
            pinned_sn = memory.get("active_order_sn")
            if pinned_sn:
                pin_line = f"{SUMMARY_PINNED_SN_PREFIX}{pinned_sn}"
                if pin_line not in rendered:
                    rendered = f"{pin_line}\n{rendered}"
            # P1 修复: 只校验 SN。active_order_id 是数据库主键，LLM 不会把它
            # 写进自然语言事实文本；而下一轮退款工具本来就会按 SN 反查 ID，
            # 不需要在摘要里同时出现主键。
            required = [
                str(value) for value in (memory.get("active_order_sn"),)
                if value is not None
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
