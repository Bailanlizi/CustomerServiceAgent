# app/api/v1/chat.py
import asyncio
import json
from types import SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from app.api.v1.schemas import ChatHistoryMessage, ChatRequest, ChatSessionResponse
from app.conversation.state_manager import (
    ConversationForbiddenError,
    ConversationMismatchError,
    ConversationNotFoundError,
    ConversationStateManager,
    default_working_memory,
)
from app.core.security import get_current_user_id
from app.services.policy_answer_guard import INTERNAL_LLM_TAG, POLICY_GUARD_TAG

router = APIRouter()
conversation_manager = ConversationStateManager()

# P2 政策回答分段推送参数：每片 ~20 字，片间 20ms 让前端逐字渲染。
# 总延迟影响可忽略（30 字答案 → 2 片 → +20ms）。
_FALLBACK_CHUNK_SIZE = 20
_FALLBACK_CHUNK_INTERVAL = 0.02


def _refund_placeholder_answer(values: dict) -> str:
    """根据 REFUND FSM workflow_stage 派生兜底回答,保持对话历史连续。"""
    stage = values.get("workflow_stage")
    if stage in {"DONE", "REJECTED"}:
        return (
            "当前退款流程已结束。如需办理新的退款，请重新说明订单号与原因。"
        )
    if stage == "WAITING_CONFIRMATION":
        return "请在前端点击「确认提交」按钮以确认退款申请。"
    if stage in {"IDENTIFY_ORDER", "COLLECT_REASON", "ELIGIBILITY_CHECKED", "SUBMITTED"}:
        return "（系统暂未生成回复，请重试或补充信息）"
    return "（系统暂未生成回复）"


@router.get("/chat/session", response_model=ChatSessionResponse)
async def get_chat_session(current_user_id: int = Depends(get_current_user_id)):
    """Return the authenticated user's default conversation and recent transcript."""
    conversation = await conversation_manager.resolve_session(
        current_user_id, client_session_id=f"account:{current_user_id}"
    )
    messages = await conversation_manager.get_messages(
        conversation.conversation_id, current_user_id, limit=100
    )
    return ChatSessionResponse(
        conversation_id=conversation.conversation_id,
        messages=[ChatHistoryMessage(role=item.role, content=item.content) for item in messages],
    )

@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user_id: int = Depends(get_current_user_id)
):
    """
    聊天接口：支持订单查询和政策咨询
    
    - ORDER:  查询用户自己的订单
    - POLICY: 从知识库检索政策信息
    """
    # 在函数内部导入，避免模块加载顺序问题
    from app.graph.workflow import app_graph
    
    if app_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not fully initialized. Please try again in a moment."
        )

    client_session_id = request.resolved_client_session_id
    try:
        if hasattr(app_graph, "aget_state"):
            conversation = await conversation_manager.resolve_session(
                current_user_id, client_session_id, request.conversation_id
            )
            memory = await conversation_manager.prepare_turn(conversation, request.question)
        else:
            # Lightweight graph doubles used by unit tests do not own persistence.
            conversation = SimpleNamespace(
                conversation_id=request.conversation_id or client_session_id,
                client_session_id=client_session_id,
                checkpoint_thread_id=f"test:{current_user_id}:{client_session_id}",
            )
            memory = default_working_memory()
        # P1: 前端「确认提交」按钮回传 user_confirmed=True；写入 working memory 顶层
        # 与 collected_slots，由子图节点读 user_confirmed 推进到 SUBMITTED 阶段。
        # prepare_turn 后续会自动把 collected_slots.user_confirmed 提升到顶层，
        # 这里同时显式写入确保 prepare_turn 之后的状态正确。
        if request.user_confirmed:
            slots = dict(memory.get("collected_slots") or {})
            slots["user_confirmed"] = True
            memory["collected_slots"] = slots
            memory["user_confirmed"] = True
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    except ConversationForbiddenError as exc:
        raise HTTPException(status_code=403, detail="Conversation belongs to another user") from exc
    except ConversationMismatchError as exc:
        raise HTTPException(status_code=409, detail="Conversation and client session do not match") from exc

    async def event_generator():
        """SSE 流式响应生成器"""
        from app.graph.workflow import app_graph
        
        thread_id = conversation.checkpoint_thread_id
        config: RunnableConfig = {"configurable": {"thread_id":  thread_id}}

        initial_state = {
            "question":  request.question,
            "user_id": current_user_id,
            "thread_id": thread_id,
            "conversation_id": str(conversation.conversation_id),
            "client_session_id": client_session_id,
            "context": [],
            "order_data": None,
            "answer": "",
            # messages 使用 add reducer；每轮显式追加本轮用户消息，供退款 Tool Agent 消费。
            "messages": [HumanMessage(content=request.question)],
            **memory,
        }

        try:
            session_payload = json.dumps({
                "type": "session",
                "conversation_id": str(conversation.conversation_id),
                "client_session_id": client_session_id,
            }, ensure_ascii=False)
            yield f"data: {session_payload}\n\n"
            token_sent = False
            fallback_answer = ""
            streamed_answer = ""
            async for event in app_graph.astream_events(
                initial_state, config, version="v2"
            ):
                kind = event["event"]
                
                # 只处理 LLM 流式输出
                if kind == "on_chat_model_stream":
                    # 内部结构化 LLM（意图识别、政策生成等）的中间 token 绝不直接输出：
                    # 既要防止未校验的 JSON 片段泄露，也要避免误置 token_sent 吞掉真正的答案。
                    tags = event.get("tags", [])
                    if POLICY_GUARD_TAG in tags or INTERNAL_LLM_TAG in tags:
                        continue
                    data = event.get("data")
                    if data and isinstance(data, dict):
                        chunk = data.get("chunk")
                        if chunk:
                            content = chunk.content
                            if content:
                                payload = json.dumps({"token": content}, ensure_ascii=False)
                                yield f"data: {payload}\n\n"
                                token_sent = True
                                streamed_answer += content

                # astream 正常会发送 on_chat_model_stream；保留节点结果兜底，
                # 兼容不发送 token 事件的 OpenAI 兼容网关。
                elif kind == "on_chain_end" and event.get("name") in {"generate", "refund_agent", "order_workflow"}:
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and output.get("answer"):
                        fallback_answer = output["answer"]

            if fallback_answer and not token_sent:
                # P2: 政策回答 / 退款话术 / 订单回复在此前已通过 PolicyAnswerGuard
                # 引用校验或 FSM 确定性模板生成，可安全推送。整段一次性 yield 会让
                # 用户在政策问答场景等 ~2.7s 才看到首字；这里按 ~20 字切片逐片
                # 推送，TTFT 感知改善，总延迟不变。SSE 事件格式与流式路径一致，
                # 前端按 token 拼接逻辑已存在。
                for i in range(0, len(fallback_answer), _FALLBACK_CHUNK_SIZE):
                    chunk = fallback_answer[i:i + _FALLBACK_CHUNK_SIZE]
                    payload = json.dumps({"token": chunk}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    await asyncio.sleep(_FALLBACK_CHUNK_INTERVAL)
                token_sent = True

            final_answer = streamed_answer if token_sent else fallback_answer

            # P1: SSE 流结束时发出 stage 字段，让前端能感知当前 refund FSM 阶段。
            # 非 REFUND 领域时 stage 为 null，前端忽略即可。
            if hasattr(app_graph, "aget_state"):
                snapshot = await app_graph.aget_state(config)
                values = dict(snapshot.values)
                stage_payload = json.dumps({
                    "type": "stage",
                    "workflow_stage": values.get("workflow_stage"),
                }, ensure_ascii=False)
                yield f"data: {stage_payload}\n\n"
                compacted = await conversation_manager.persist_turn(conversation, values)
                # P1 修复: REFUND 域若 answer 为空(终端状态短路 / 资格预检空响应等),
                # 也写入由 workflow_stage 派生的占位 assistant 消息,保持历史连续。
                # 非 REFUND 域继续仅在有真实回答时落库。
                persist_answer = final_answer
                if not persist_answer and values.get("active_domain") == "REFUND":
                    persist_answer = _refund_placeholder_answer(values)
                if persist_answer:
                    await conversation_manager.append_messages(
                        conversation.conversation_id,
                        current_user_id,
                        request.question,
                        persist_answer,
                    )
                messages = list(values.get("messages") or [])
                retained = conversation_manager.retained_messages(messages)
                if compacted and len(retained) < len(messages):
                    from langchain_core.messages import RemoveMessage
                    from langgraph.graph.message import REMOVE_ALL_MESSAGES
                    await app_graph.aupdate_state(
                        config,
                        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *retained]},
                    )

            yield "data: [DONE]\n\n"
            
        except Exception as e:  # noqa: BLE001 - SSE must serialize all runtime failures.
            error_msg = json.dumps({'error': str(e)}, ensure_ascii=False)
            yield f"data: {error_msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
