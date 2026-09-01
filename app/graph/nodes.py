# app/graph/nodes.py
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.core.database import async_session_maker
from app.models.knowledge import KnowledgeChunk
from app.models.order import Order
from app.graph.state import AgentState
from sqlmodel import select
from pydantic import BaseModel, SecretStr
from langchain_core.messages import AIMessage, BaseMessage
from app.graph.tools import refund_tools
from app.services.policy_retrieval import QwenEmbeddings, embedding_model, retrieve_policy


# 相似度阈值：只有距离 < 0.5 才认为相关
SIMILARITY_THRESHOLD = 0.5

# ==========================================
# 全局组件初始化
# ==========================================

# 1. LLM 模型 (用于生成回答)
llm = ChatOpenAI(
    base_url=settings.OPENAI_BASE_URL,
    api_key=SecretStr(settings.OPENAI_API_KEY),
    model=settings.LLM_MODEL,
    temperature=0 
)

# 2. Prompt 模板
PROMPT_TEMPLATE = """
你是一个专业的电商政策咨询专家。请基于以下检索到的 context 回答用户的问题。

规则：
1. 只能依据 context 中的信息回答。
2. 如果 context 为空或没有相关信息，请直接回答"抱歉，暂未查询到相关规定"，严禁编造。
3. 语气专业、客气。

Context: 
{context}

User Question: 
{question}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ==========================================
# 节点函数定义
# ==========================================

async def retrieve(state: AgentState) -> dict:
    """
    检索节点：带阈值过滤的硬逻辑
    """
    question = state["question"]
    print(f"🔍 [Retrieve] 正在检索: {question}")

    retrieved = await retrieve_policy(question, similarity_threshold=SIMILARITY_THRESHOLD)
    valid_chunks = [chunk.content for chunk in retrieved]
    for chunk in retrieved:
        print(f"   - {chunk.clause_ids or ['未标注条款']}: {chunk.content[:10]}... | 距离分: {chunk.distance:.4f}")

    print(f" [Retrieve] 最终有效记录: {len(valid_chunks)} 条")
    return {"context": valid_chunks}


# Generate 节点的 System Prompt
GENERATE_SYSTEM_PROMPT = """
你是一个电商客服助手。请根据提供的 [参考信息] 友好地回答用户。

规则：
1. 如果是订单信息，请清晰列出订单号、状态、总额和配送地址。
2. 如果是政策信息，请引用相关条款。
3. 如果参考信息为空，请礼貌地告知无法查到，并引导用户提供更多细节（如单号）。
4. 严禁编造数据库中不存在的订单状态。
"""

async def generate(state: AgentState) -> dict:
    print(" [Generate] 正在生成综合回复...")
    
    # 1. 组装参考信息
    context_parts = []
    
    # 加入政策背景
    if state.get("context"):
        context_parts.append("【相关政策】:\n" + "\n".join(state["context"]))
    
    # 加入订单背景
    if state.get("order_data"):
        order_raw = state["order_data"]
        if hasattr(order_raw, "model_dump"):
            order = order_raw.model_dump()
        else:
            order = order_raw or {}

        def safe_get(d, *keys, default=None):
            if not isinstance(d, dict):
                return default
            for k in keys:
                if k in d and d[k] is not None: 
                    return d[k]
            return default

        order_sn = safe_get(order, "order_sn", "sn", default="未知")
        status = safe_get(order, "status", default="未知")
        amount = safe_get(order, "total_amount", "amount", default=0)
        tracking = safe_get(order, "tracking_number", "tracking", "shipping_address", default=None)
        items = safe_get(order, "items", default=[])

        order_str = (
            f"【订单详情】:\n"
            f"- 订单号: {order_sn}\n"
            f"- 当前状态: {status}\n"
            f"- 订单金额: {amount} 元\n"
            f"- 收货地址: {tracking or '暂无'}\n"
            f"- 商品明细:  {items}"
        )
        context_parts.append(order_str)

    context_info = "\n\n".join(context_parts) if context_parts else "暂无相关参考信息。"

    # 2. 构建用户消息
    user_content = f"""[参考信息]：
{context_info}

[用户问题]：
{state['question']}"""

    # 3. 调用 LLM
    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ]
    
    response = None
    async for chunk in llm.astream(messages):
        response = chunk if response is None else response + chunk

    return {"answer": response.content if response else ""}


# 意图识别的 System Prompt
INTENT_PROMPT = """你是一个电商客服分类器。你的任务是根据用户的输入，将其归类为以下四种意图之一：

- "ORDER":   用户询问关于他们自己的订单状态、物流、详情等（但不是退货）。
  示例："我的订单到哪了？"、"查询订单 SN20240001"

- "POLICY":  用户询问关于平台通用的退换货、运费、时效等政策信息。
  示例："内衣可以退货吗？"、"运费怎么算？"

- "REFUND": 用户明确表示要办理退货、退款、换货等售后服务。
  示例："我要退货"、"申请退款"、"这个订单我不要了"

- "OTHER": 用户进行闲聊、打招呼或提出与上述无关的问题。
  示例："你好"、"讲个笑话"

只返回分类标签（ORDER/POLICY/REFUND/OTHER），不要返回任何其他文字。"""


class IntentDecision(BaseModel):
    """受限的意图路由结果，避免依赖模型的自由文本输出。"""

    intent: Literal["ORDER", "POLICY", "REFUND", "OTHER"]


intent_classifier = llm.with_structured_output(IntentDecision)


async def intent_router(state: AgentState):
    """
    意图识别节点：判断用户想干什么
    """
    print(f" [Router] 正在分析意图:  {state['question']}")
    
    decision = await intent_classifier.ainvoke([
        SystemMessage(content=INTENT_PROMPT),
        HumanMessage(content=state["question"])
    ])
    intent = decision.intent
        
    print(f" [Router] 识别结果: {intent}")
    return {"intent": intent}

async def query_order(state: AgentState):
    """
    订单查询节点：从数据库查数据
    """
    question = state["question"]
    user_id = state["user_id"]
    
    import re
    order_sn_match = re.search(r'SN\d+', question.upper())
    
    # 构造查询
    if not order_sn_match: 
        print(" [QueryOrder] 获取用户最近订单")
        stmt = (
            select(Order)
            .where(Order.user_id == user_id)
            .order_by(Order.created_at.desc())
            .limit(1)
        )
    else:
        order_sn = order_sn_match.group()
        print(f" [QueryOrder] 查询订单号: {order_sn}")
        stmt = select(Order).where(
            Order.order_sn == order_sn,
            Order.user_id == user_id 
        )

    async with async_session_maker() as session:
        result = await session.exec(stmt)
        order = result.first()

    if not order:
        return {
            "order_data": None, 
            "context": ["用户询问了订单，但数据库中未查到相关记录。"]
        }
    
    # 组装订单信息
    items_str = ", ".join([f"{i['name']}(x{i['qty']})" for i in order.items])
    order_context = (
        f"订单号: {order.order_sn}\n"
        f"状态: {order.status}\n"
        f"商品:  {items_str}\n"
        f"金额: {order.total_amount}元\n"
        f"物流单号: {order.tracking_number or '暂无'}"
    )
    
    return {
        "order_data":  order.model_dump(), 
        "context": [order_context]
    }


REFUND_AGENT_PROMPT = """你是电商售后助手。必须使用工具获取或改变退款数据，不能编造任何订单、申请或审核结果。
根据用户请求选择一个工具：资格预检用 check_refund_eligibility；明确申请退款/退货用 submit_refund_application；查询进度用 query_refund_status。
如果缺少工具所需的订单号、退款原因或申请编号，请直接向用户索要该信息，不要调用工具。工具返回后，用简洁中文说明结果。"""


async def refund_agent(state: AgentState) -> dict:
    """退款 Agent：模型自主选择受控工具，身份信息由 ToolNode 注入。"""
    messages: List[BaseMessage] = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=state["question"])]

    response = None
    async for chunk in llm.bind_tools(refund_tools).astream(
        [SystemMessage(content=REFUND_AGENT_PROMPT), *messages]
    ):
        response = chunk if response is None else response + chunk

    if response is None:
        return {"answer": "抱歉，暂时无法处理退款请求，请稍后重试。"}
    result: dict = {"messages": [response]}
    if not response.tool_calls:
        result["answer"] = response.content
    return result


def should_call_refund_tool(state: AgentState) -> str:
    """仅在最新模型消息携带工具调用时进入 ToolNode。"""
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "refund_tools"
    return "done"
