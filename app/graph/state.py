# app/graph/state.py
import operator
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # 基础信息
    question: str
    user_id: int  
    
    # 意图标签:  "POLICY" 或 "ORDER" 或 "REFUND" 或 "OTHER"
    intent: str | None
    
    # 历史记录 (用于多轮对话)
    history:  Annotated[list[dict], operator.add]
    
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
    
    # v4.0 新增：结构化消息列表
    messages:  Annotated[list[BaseMessage], operator.add]
    
    # 最终回复
    answer: str
