"""Parse LLM streaming output into thinking / answer segments.

把原本混在 task_service._stream_answer 里的"边读流边切分思考/回答"逻辑
抽出来，用一个纯函数式接口驱动。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)


ANSWER_MARKS = ("答案：", "答案:")
THINK_MARKS = ("思考：", "思考:")


async def async_iter_sync(iterable, sentinel=object()):
    """把同步 generator 转成 async generator（每次 next 都走线程池）。"""
    iterator = iter(iterable)

    def _next():
        return next(iterator, sentinel)

    while True:
        item = await asyncio.to_thread(_next)
        if item is sentinel:
            break
        yield item


def parse_provider_chunk(chunk: Any) -> Iterable[Dict[str, Any]]:
    """解析单个 provider chunk（可能是 SSE 行，可能是纯文本）。"""
    if not isinstance(chunk, str):
        return []

    # 分支 1：SSE 格式
    if "data:" in chunk:
        events: List[Dict[str, Any]] = []
        for line in chunk.splitlines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    # 分支 2：纯文本
    stripped = chunk.strip()
    if not stripped:
        return []
    if stripped in ("[DONE]", "data:[DONE]", "data: [DONE]"):
        return []
    logger.warning("[dsh][stream] fallback text (not SSE): %r", chunk[:200])
    return [{"type": "text", "text": chunk}]


class StreamSplitter:
    """把 LLM 流切分成 thinking / answer 两段。

    用法：
        splitter = StreamSplitter(
            on_thinking=lambda text: ...,
            on_answer=lambda text: ...,
        )
        async for chunk in async_iter_sync(provider):
            for payload in parse_provider_chunk(chunk):
                splitter.feed(payload)

        thinking_text, answer_text = splitter.finalize()
    """

    def __init__(
        self,
        on_thinking: Optional[Callable[[str], None]] = None,
        on_answer: Optional[Callable[[str], None]] = None,
    ):
        self._on_thinking = on_thinking
        self._on_answer = on_answer

        self.thinking_started = False
        self.answer_started = False
        self.thinking_buffer = ""
        self.answer_buffer = ""
        self._pending = ""

    def feed(self, payload: Dict[str, Any]) -> Optional[str]:
        """喂入一个 payload。返回 "error" 表示 payload 是错误。"""
        if payload.get("error"):
            return "error"

        text = payload.get("text") or ""
        if not isinstance(text, str) or not text:
            return None

        self._pending += text

        if not self.thinking_started:
            self._handle_thinking_start()
            # 若仍在等待 thinking 标记，直接返回
            if not self.thinking_started:
                return None

        if self.thinking_started and not self.answer_started:
            self._handle_answer_transition()

        if self.answer_started and self._pending:
            self.answer_buffer += self._pending
            self._pending = ""
            if self._on_answer:
                self._on_answer(self.answer_buffer)
        return None

    def _handle_thinking_start(self) -> None:
        """尝试从 _pending 里找 thinking 标记。"""
        matched_think = None
        for mark in THINK_MARKS:
            idx = self._pending.find(mark)
            if idx >= 0:
                matched_think = (mark, idx)
                break

        if matched_think is not None:
            mark, idx = matched_think
            self._pending = self._pending[idx + len(mark):]
            self.thinking_started = True
            if self._on_thinking:
                self._on_thinking("")
            return

        # 检测是否是标记的前缀（还在等待完整标记）
        might_be_prefix = any(
            self._pending.startswith(mark[:i]) or mark[:i].startswith(self._pending[:i])
            for mark in THINK_MARKS + ANSWER_MARKS
            for i in range(1, min(len(mark), len(self._pending)) + 1)
        ) and len(self._pending) < 4
        if might_be_prefix:
            return

        # 没找到 thinking 标记 → 认为直接进入 thinking（无显式标记）
        self.thinking_started = True
        if self._on_thinking:
            self._on_thinking("")

    def _handle_answer_transition(self) -> None:
        """在 thinking 已开始、answer 未开始时，找 answer 标记。"""
        matched_mark = None
        matched_idx = -1
        for mark in ANSWER_MARKS:
            idx = self._pending.find(mark)
            if idx >= 0 and (matched_idx < 0 or idx < matched_idx):
                matched_mark = mark
                matched_idx = idx

        if matched_mark is not None:
            thinking_part = self._pending[:matched_idx]
            if thinking_part:
                self.thinking_buffer += thinking_part
                if self._on_thinking:
                    self._on_thinking(self.thinking_buffer)
            self._pending = self._pending[matched_idx + len(matched_mark):]
            self.answer_started = True
            if self._on_answer:
                self._on_answer("")
            return

        # 保留末尾可能是标记前缀的部分
        keep = 0
        for mark in ANSWER_MARKS:
            for i in range(1, len(mark)):
                if self._pending.endswith(mark[:i]):
                    keep = max(keep, i)
        if keep:
            thinking_part, self._pending = self._pending[:-keep], self._pending[-keep:]
        else:
            thinking_part, self._pending = self._pending, ""

        if thinking_part:
            self.thinking_buffer += thinking_part
            if self._on_thinking:
                self._on_thinking(self.thinking_buffer)

    def finalize(self) -> Tuple[str, str]:
        """流结束时调用。返回 (thinking_text, answer_text)。"""
        # 剩余 pending 的处理
        if not self.thinking_started and self._pending:
            self.answer_buffer = self._pending
            self._pending = ""
            if self._on_answer:
                self._on_answer(self.answer_buffer)
        elif self.thinking_started and not self.answer_started and self._pending:
            self.thinking_buffer += self._pending
            self._pending = ""
            if self._on_thinking:
                self._on_thinking(self.thinking_buffer)

        # 只有 thinking 没有 answer → 把 thinking 当 answer
        if not self.answer_buffer and self.thinking_buffer and not self.answer_started:
            self.answer_buffer = self.thinking_buffer
            self.thinking_buffer = ""

        return self.thinking_buffer, self.answer_buffer