# app/graph/state.py
from typing import TypedDict, List, Optional, Annotated, NotRequired
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # 基础信息
    question: str
    user_id: int  
    
    # 意图标签:  "POLICY" 或 "ORDER" 或 "REFUND" 或 "OTHER"
    intent: Optional[str]
    
    # 历史记录 (用于多轮对话)
    history:  Annotated[List[dict], operator.add]
    
    # 检索到的知识 
    context:  List[str]

    # 仅政策咨询路径注入的文档级优先级规则，不参与向量 Top-K 竞争。
    policy_rules: NotRequired[List[str]]
    
    # 查到的订单数据 
    order_data: Optional[dict]
    
    # v4.0 新增：会话 ID
    thread_id: str
    
    # v4.0 新增：结构化消息列表
    messages:  Annotated[List[BaseMessage], operator.add]
    
    # 最终回复
    answer: str
