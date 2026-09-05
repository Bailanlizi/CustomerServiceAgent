# app/graph/state.py
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 基础信息
    question: str
    user_id: int  
    
    # 意图标签:  "POLICY" 或 "ORDER" 或 "REFUND" 或 "OTHER"
    intent: str | None

    # 检索到的知识
    context:  list[str]

    # 仅政策咨询路径注入的文档级优先级规则，不参与向量 Top-K 竞争。
    policy_rules: NotRequired[list[str]]

    # 政策检索的完整、JSON 可序列化证据；供生成与引用审计使用。
    policy_evidence: NotRequired[list[dict]]

    # 政策回答的引用校验结果，随 LangGraph checkpoint 保存。
    policy_answer_audit: NotRequired[dict]
    
    # 查到的订单数据 
    order_data: dict | None
    
    # v4.0 新增：会话 ID
    thread_id: str

    # P0: stable business session and structured working memory
    conversation_id: NotRequired[str]
    client_session_id: NotRequired[str]
    active_domain: NotRequired[Literal["ORDER", "POLICY", "REFUND"] | None]
    active_order_id: NotRequired[int | None]
    active_order_sn: NotRequired[str | None]
    conversation_goal: NotRequired[str | None]
    collected_slots: NotRequired[dict[str, object]]
    pending_slots: NotRequired[list[str]]
    workflow_stage: NotRequired[str | None]
    last_tool_result: NotRequired[dict | None]
    next_action: NotRequired[Literal["ASK", "TOOL_CALL", "COMPOSE", "WAIT"] | None]
    conversation_summary: NotRequired[str | None]
    # P1 修复: 退款原因分类枚举，由 SlotExtraction 一次产出，写入工作记忆，
    # 提交退款工具通过 InjectedState 直接消费，避免 LLM 二次映射。
    refund_reason_category: NotRequired[str | None]
    # P1: 用户对退款申请的显式确认标志。前端按钮回传 user_confirmed=True；
    # prepare_turn 从 collected_slots.user_confirmed 同步到顶层，让退款工具通过
    # InjectedState("user_confirmed") 直接校验；persist_turn 反向同步回 collected_slots，
    # 保证持久化真源仍为 collected_slots。
    user_confirmed: NotRequired[bool | None]
    
    # v4.0 新增：结构化消息列表
    messages: Annotated[list[BaseMessage], add_messages]
    
    # 最终回复
    answer: str
