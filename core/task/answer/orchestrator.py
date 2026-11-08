"""answer_stream：原 _stream_answer 的主入口串联。"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from core.task.stream_parser import StreamSplitter

from .asset_resolve import resolve_assets
from .fallback_search import run_fallback, should_fallback
from .multimodal import is_multimodal  # noqa: F401
from .prompt_build import build_answer_messages, inject_memory_intent
from .provider_call import select_provider
from .splitter_bridge import feed_provider, make_splitter

logger = logging.getLogger(__name__)


async def answer_stream(
    *,
    task_id: str,
    session_id: str,
    prior_messages: List[Dict[str, str]],
    message: str,
    mode: str,
    attachments: List[Dict[str, Any]],
    turn_id: str,
    rag_context: Optional[str],
    rag_resources: Optional[List[Dict[str, Any]]],
    tool_context: Optional[List[str]],
    make_step: Callable[..., Dict[str, Any]],
    emit_step: Callable[[str, Dict[str, Any]], str],
    session_provider,
    rag_provider,
    llm_provider,
    multimodal_provider,
    agent_provider,
    persist_task_status: Callable[[str, str], None],
) -> AsyncIterator[str]:
    from core.task.service import sse_event

    thinking_step_id = f"{task_id}:{turn_id}:thinking"
    answer_step_id = f"{task_id}:{turn_id}:answer"

    yield emit_step(
        task_id,
        make_step(thinking_step_id, "思考过程", "running", "think", log=""),
    )

    tool_context = inject_memory_intent(session_id, tool_context)

    if rag_resources:
        yield emit_step(
            task_id,
            make_step(
                f"{task_id}:{turn_id}:rag_resources",
                "引用知识库 " + str(len(rag_resources)) + " 个资源",
                "done",
                "resource",
                resources=rag_resources,
            ),
        )

    is_multi = bool(attachments and all(
        is_multimodal(a) for a in attachments
    ))
    if is_multi:
        provider, _ = select_provider(
            attachments=attachments,
            message=message,
            messages=None,
            mode=mode,
            multimodal_provider=multimodal_provider,
            llm_provider=llm_provider,
        )
    else:
        messages = build_answer_messages(
            message=message,
            mode=mode,
            attachments=attachments,
            rag_context=rag_context,
            tool_context=tool_context,
            prior_messages=prior_messages,
            task_id=task_id,
            session_provider=session_provider,
        )
        provider, _ = select_provider(
            attachments=attachments,
            message=message,
            messages=messages,
            mode=mode,
            multimodal_provider=multimodal_provider,
            llm_provider=llm_provider,
        )

    splitter, drain, answer_step_id, thinking_step_id = make_splitter(
        task_id=task_id,
        turn_id=turn_id,
        make_step=make_step,
        emit_step=emit_step,
    )

    async for evt in feed_provider(provider, splitter, drain):
        yield evt

    thinking_text, answer_text = splitter.finalize()
    for evt in drain():
        yield evt

    yield emit_step(
        task_id,
        make_step(
            thinking_step_id, "思考过程", "done", "think", log=thinking_text
        ),
    )

    if not answer_text:
        answer_text = "未生成有效回答。"

    if should_fallback(mode, answer_text, tool_context):
        logger.info("[dsh][answer] answer admits missing; triggering fallback")
        extra = await run_fallback(message, agent_provider)
        if extra:
            retry_context = list(tool_context or []) + extra
            async for evt in answer_stream(
                task_id=task_id,
                session_id=session_id,
                prior_messages=prior_messages,
                message=message,
                mode=mode,
                attachments=attachments,
                turn_id=turn_id,
                rag_context=None,
                rag_resources=None,
                tool_context=retry_context,
                make_step=make_step,
                emit_step=emit_step,
                session_provider=session_provider,
                rag_provider=rag_provider,
                llm_provider=llm_provider,
                multimodal_provider=multimodal_provider,
                agent_provider=agent_provider,
                persist_task_status=persist_task_status,
            ):
                yield evt
            return

    if rag_resources:
        answer_text = resolve_assets(answer_text, rag_resources, rag_provider)

    yield emit_step(
        task_id,
        make_step(answer_step_id, "Agent 回复", "done", "answer", log=answer_text),
    )

    try:
        from infra.debug_log.llm import dump_llm_call
        dump_llm_call(
            kind="stream_answer_response",
            prompt="",
            response=answer_text,
            extra={
                "thinking_len": len(thinking_text or ""),
                "answer_len": len(answer_text or ""),
            },
        )
    except Exception:
        logger.exception("dump stream_answer response failed")

    session_provider().add_message(session_id, "assistant", answer_text)
    persist_task_status(task_id, "done")
    yield sse_event("done", task_status="done")