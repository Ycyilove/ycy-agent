"""StreamSplitter ↔ SSE step 事件桥接。"""
from __future__ import annotations

from collections import deque
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from core.task.stream_parser import (
    StreamSplitter,
    async_iter_sync,
    parse_provider_chunk,
)


def make_splitter(
    *,
    task_id: str,
    turn_id: str,
    make_step: Callable[..., Dict[str, Any]],
    emit_step: Callable[[str, Dict[str, Any]], str],
):
    """返回 (splitter, drain_events, on_error)。"""
    answer_step_id = f"{task_id}:{turn_id}:answer"
    thinking_step_id = f"{task_id}:{turn_id}:thinking"
    events: "deque[Dict[str, Any]]" = deque()

    def on_thinking(text: str) -> None:
        step = make_step(
            thinking_step_id, "思考过程", "running", "think", log=text
        )
        events.append({"kind": "thinking", "step": step})

    def on_answer(text: str) -> None:
        step = make_step(
            answer_step_id, "正在生成回答…", "running", "answer", log=text
        )
        events.append({"kind": "answer", "step": step})

    splitter = StreamSplitter(on_thinking=on_thinking, on_answer=on_answer)

    def drain() -> List[str]:
        out = []
        while events:
            evt = events.popleft()
            out.append(emit_step(task_id, evt["step"]))
        return out

    return splitter, drain, answer_step_id, thinking_step_id


async def feed_provider(
    provider,
    splitter: StreamSplitter,
    drain: Callable[[], List[str]],
) -> AsyncIterator[str]:
    """逐 chunk 喂 splitter，yield 出 SSE。"""
    from core.task.service import sse_event

    async for chunk in async_iter_sync(provider):
        for payload in parse_provider_chunk(chunk):
            result = splitter.feed(payload)
            if result == "error":
                yield sse_event(
                    "error",
                    message=payload.get("text") or "生成回答失败",
                )
            for evt in drain():
                yield evt