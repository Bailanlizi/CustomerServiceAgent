# app/api/v1/chat.py
import json
from types import SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from app.api.v1.schemas import ChatRequest
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

                # astream 正常会发送 on_chat_model_stream；保留节点结果兜底，
                # 兼容不发送 token 事件的 OpenAI 兼容网关。
                elif kind == "on_chain_end" and event.get("name") in {"generate", "refund_agent"}:
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and output.get("answer"):
                        fallback_answer = output["answer"]

            if fallback_answer and not token_sent:
                payload = json.dumps({"token": fallback_answer}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

            if hasattr(app_graph, "aget_state"):
                snapshot = await app_graph.aget_state(config)
                values = dict(snapshot.values)
                compacted = await conversation_manager.persist_turn(conversation, values)
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
