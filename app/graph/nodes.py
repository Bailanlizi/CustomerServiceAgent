# app/graph/nodes.py
import asyncio
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, ValidationError
from sqlmodel import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.graph.state import AgentState
from app.graph.tools import refund_tools
from app.models.order import Order
from app.services.policy_answer_guard import (
    INTERNAL_LLM_TAG,
    NO_EVIDENCE_FALLBACK,
    POLICY_GUARD_TAG,
    SAFE_POLICY_FALLBACK,
    PolicyAnswer,
    PolicyCitationValidationError,
    allowed_evidence_ids,
    validate_policy_answer,
)
from app.services.policy_retrieval import load_policy_rules, retrieve_policy

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

POLICY_GENERATE_SYSTEM_PROMPT = """
你是电商客服的政策回答助手。只能依据提供的政策证据回答，不得补充、猜测或承诺证据中不存在的内容。

请使用给定的结构化输出：
1. answer 是面向用户的自然语言回答，不要在其中展示条款编号。
2. applied_clause_ids 填直接用于结论的条款编号。
3. evidence_clause_ids 填支撑解释的全部条款编号，必须包含 applied_clause_ids。
4. 两个条款列表只能使用“允许引用编号”中的值。
5. 若没有足够证据，请明确说明暂无法依据当前政策作出确定承诺，并返回空的条款列表。
"""

# ==========================================
# 节点函数定义
# ==========================================

async def retrieve(state: AgentState) -> dict:
    """
    检索节点：带阈值过滤的硬逻辑
    """
    question = state["question"]
    print(f"🔍 [Retrieve] 正在检索: {question}")

    retrieved, policy_rules = await asyncio.gather(
        retrieve_policy(question, similarity_threshold=SIMILARITY_THRESHOLD),
        load_policy_rules(),
    )
    policy_evidence = [
        {
            "content": chunk.content,
            "source": chunk.source,
            "clause_ids": chunk.clause_ids,
            "canonical_clause_ids": chunk.canonical_clause_ids,
            "source_type": chunk.source_type,
            "rank": chunk.rank,
            "distance": chunk.distance,
        }
        for chunk in retrieved
    ]
    # 保留旧字段，避免订单与非政策生成路径发生行为变化。
    valid_chunks = [item["content"] for item in policy_evidence]
    for chunk in retrieved:
        print(f"   - {chunk.clause_ids or ['未标注条款']}: {chunk.content[:10]}... | 距离分: {chunk.distance:.4f}")

    print(f" [Retrieve] 最终有效记录: {len(valid_chunks)} 条")
    return {
        "context": valid_chunks,
        "policy_evidence": policy_evidence,
        "policy_rules": policy_rules,
    }


# Generate 节点的 System Prompt
GENERATE_SYSTEM_PROMPT = """
你是一个电商客服助手。请根据提供的 [参考信息] 友好地回答用户。

规则：
1. 如果是订单信息，请清晰列出订单号、状态、总额和配送地址。
2. 如果是政策信息，请引用相关条款。
3. 如果参考信息为空，请礼貌地告知无法查到，并引导用户提供更多细节（如单号）。
4. 严禁编造数据库中不存在的订单状态。
"""

policy_answer_llm = llm.with_structured_output(PolicyAnswer).with_config(
    {"tags": [POLICY_GUARD_TAG]}
)


def _format_policy_evidence(evidence: list[dict[str, Any]]) -> str:
    """将结构化检索结果呈现给模型，同时保留合法 ID 的可见边界。"""
    parts = []
    for item in evidence:
        ids = list(item.get("clause_ids", [])) + list(item.get("canonical_clause_ids", []))
        parts.append(
            f"【来源：{item.get('source', '未知')} | 类型：{item.get('source_type', 'policy')} | "
            f"允许编号：{', '.join(ids) or '无'}】\n{item.get('content', '')}"
        )
    return "\n\n".join(parts) or "暂无相关参考信息。"


RETRY_FEEDBACK_TEMPLATE = (
    "你上一次的回答未通过引用校验，错误原因：{error}\n"
    "请重新生成：applied_clause_ids 与 evidence_clause_ids 只能使用[允许引用编号]中的值，"
    "applied_clause_ids 必须是 evidence_clause_ids 的子集，且 answer 中不得出现任何条款编号。"
)


async def _generate_verified_policy_answer(state: AgentState) -> dict:
    """生成政策回答；仅在引用校验通过后将答案交给 API 输出。"""
    evidence: list[dict[str, Any]] = list(state.get("policy_evidence", []))

    # 无检索证据时不再让模型自由生成：直接返回确定性安全答复，
    # 从源头杜绝“检索不到 → 仍作出承诺”的过度承诺路径（空证据下引用校验会全部空转）。
    if not evidence:
        return {
            "answer": NO_EVIDENCE_FALLBACK,
            "policy_answer_audit": {
                "status": "no_evidence",
                "applied_clause_ids": [],
                "evidence_clause_ids": [],
                "allowed_evidence_ids": [],
                "retry_count": 0,
            },
        }

    allowed_ids = allowed_evidence_ids(evidence)
    policy_rules = state.get("policy_rules", [])
    context = _format_policy_evidence(evidence)
    if policy_rules:
        context += "\n\n【政策适用优先级背景】\n" + "\n".join(policy_rules)

    messages = [
        SystemMessage(content=POLICY_GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"[政策证据]\n{context}\n\n"
            f"[允许引用编号]\n{', '.join(allowed_ids) or '无'}\n\n"
            f"[用户问题]\n{state['question']}"
        )),
    ]

    last_error: str | None = None
    for attempt in range(2):
        attempt_messages = list(messages)
        if last_error is not None:
            # 把校验失败的具体原因反馈给模型，提高重试成功率。
            attempt_messages.append(
                HumanMessage(content=RETRY_FEEDBACK_TEMPLATE.format(error=last_error))
            )
        try:
            raw_answer = await policy_answer_llm.ainvoke(attempt_messages)
            candidate = PolicyAnswer.model_validate(raw_answer)
            verified = validate_policy_answer(
                candidate,
                allowed_ids,
                evidence_present=True,
            )
            return {
                "answer": verified.answer,
                "policy_answer_audit": {
                    "status": "passed",
                    "applied_clause_ids": verified.applied_clause_ids,
                    "evidence_clause_ids": verified.evidence_clause_ids,
                    "allowed_evidence_ids": allowed_ids,
                    "retry_count": attempt,
                },
            }
        except (PolicyCitationValidationError, ValidationError) as exc:
            last_error = str(exc)
            print(f" [PolicyGuard] 第 {attempt + 1} 次引用校验失败: {exc}")
        except Exception as exc:  # noqa: BLE001 - 任何解析异常都必须阻止未校验回答输出。
            # 结构化响应解析失败时同样不得回退到未经校验的自由文本。
            last_error = f"{type(exc).__name__}: {exc}"
            print(f" [PolicyGuard] 第 {attempt + 1} 次结构化生成失败: {last_error}")

    return {
        "answer": SAFE_POLICY_FALLBACK,
        "policy_answer_audit": {
            "status": "fallback",
            "applied_clause_ids": [],
            "evidence_clause_ids": [],
            "allowed_evidence_ids": allowed_ids,
            "retry_count": 1,
        },
    }

async def generate(state: AgentState) -> dict:
    print(" [Generate] 正在生成综合回复...")

    # 政策答复必须在输出到 SSE 前完成结构化引用校验。
    if state.get("intent") == "POLICY":
        result = await _generate_verified_policy_answer(state)
        # 回写 AI 消息：保证会话历史 Human/AI 成对，否则后续退款 Agent 会看到
        # 连续多条无人回复的用户消息，从而把旧问题一并重复作答。
        result["messages"] = [AIMessage(content=result["answer"])]
        return result
    
    # 1. 组装参考信息
    context_parts = []
    
    # 加入政策背景
    if state.get("context"):
        context_parts.append("【相关政策】:\n" + "\n".join(state["context"]))

    if state.get("intent") == "POLICY" and state.get("policy_rules"):
        context_parts.append("【政策适用优先级】:\n" + "\n".join(state["policy_rules"]))
    
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

    answer = response.content if response else ""
    return {
        "answer": answer,
        # 同上：回写 AI 消息，维持 Human/AI 交替的合法会话结构。
        "messages": [AIMessage(content=answer)],
    }


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


intent_classifier = llm.with_structured_output(IntentDecision).with_config(
    {"tags": [INTERNAL_LLM_TAG]}
)


async def intent_router(state: AgentState):
    """
    意图识别节点：判断用户想干什么
    """
    # ConversationStateManager has already resolved compatible follow-ups and
    # explicit domain switches. Reuse that decision to avoid reclassifying a
    # bare order number or refund reason as an unrelated intent.
    active_domain = state.get("active_domain")
    if active_domain in {"ORDER", "POLICY", "REFUND"}:
        print(f" [Router] 复用活跃领域: {active_domain}")
        return {"intent": active_domain}

    print(f" [Router] 正在分析意图:  {state['question']}")
    
    decision = await intent_classifier.ainvoke(
        [
            SystemMessage(content=INTENT_PROMPT),
            HumanMessage(content=state["question"]),
        ]
    )
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
订单号、退款原因与分类已由系统在调用工具时从工作记忆自动注入，不要在 prompt 中重复要求用户确认这些字段；如果工具返回"缺少订单号/原因"的提示，请向用户索要该信息。
申请编号（query_refund_status 需要的 refund_id）由用户主动提供，不来自工作记忆。

回答范围约束：
1. 只回答用户最新一条消息。历史消息仅用于理解指代（如"它""这个订单"），不要重复回答已经回复过的旧问题。
2. 只处理售后范围内的请求。订单物流查询、通用政策咨询等非售后问题若历史中已回复过，直接忽略；未回复的请说明你只负责售后，不要越权作答。"""


# 退款 Agent 最多携带的历史轮数。跨意图历史（订单查询、政策咨询）无限堆积会让模型
# 把旧问题一并重复作答，这里限制为最近若干轮。
MAX_REFUND_HISTORY_TURNS = 3


def select_refund_context_messages(
    messages: list[BaseMessage], question: str
) -> list[BaseMessage]:
    """为退款 Agent 选取安全的历史窗口。

    两个约束：
    1. 只保留最近 MAX_REFUND_HISTORY_TURNS 轮，避免无关的跨意图历史干扰当前售后流程；
    2. 窗口起点必须是一条 HumanMessage——否则可能把带 tool_calls 的 AIMessage
       与其配对的 ToolMessage 拆开，触发模型 API 报错。
    """
    if not messages:
        return [HumanMessage(content=question)]

    human_indexes = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not human_indexes:
        return [HumanMessage(content=question)]

    # 从倒数第 MAX_REFUND_HISTORY_TURNS 条用户消息开始；历史不足时退化为从第一条开始。
    start = human_indexes[max(len(human_indexes) - MAX_REFUND_HISTORY_TURNS, 0)]
    return list(messages[start:])


async def refund_agent(state: AgentState) -> dict:
    """退款 Agent：模型自主选择受控工具，身份信息由 ToolNode 注入。"""
    messages: list[BaseMessage] = select_refund_context_messages(
        list(state.get("messages", [])), state["question"]
    )

    slots = state.get("collected_slots", {})
    memory_context = (
        "\n\n当前已确认的会话信息（不要重复询问）："
        f"订单号={state.get('active_order_sn') or slots.get('order_sn') or '未确认'}；"
        f"退款原因={slots.get('refund_reason') or '未确认'}；"
        f"仍缺字段={state.get('pending_slots', [])}；"
        f"摘要={state.get('conversation_summary') or '无'}。"
        "只有仍缺失的字段才可以向用户询问。"
    )
    response = None
    async for chunk in llm.bind_tools(refund_tools).astream(
        [SystemMessage(content=REFUND_AGENT_PROMPT + memory_context), *messages]
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
