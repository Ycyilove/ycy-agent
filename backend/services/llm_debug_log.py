"""把每次 LLM 调用落盘，按「会话 → 任务 → 提问」三级归类。

目录结构：
    logs/llm/
      <session_id>/
        <task_id>/
          turn-01-ask-<slug>/
            01-decide.json
            02-fill_params-read_text_file.json
            03-decide.json
            04-fill_params-edit_file.json
            05-stream_answer.json
            06-stream_answer_response.json
          turn-02-ask-<slug>/
            ...

环境变量：
    LLM_DEBUG_DUMP=1                开启（默认 1）
    LLM_DEBUG_DIR=./logs/llm        自定义根目录
    LLM_DEBUG_MAX_CHARS=100000      单字段最大字符（超出截断）

关键 API：
    begin_turn(session_id, task_id, question) -> TurnContext
        开启一个 turn，后续所有 dump_llm_call 都归到该目录。
    dump_llm_call(kind=..., prompt=..., response=...) -> Optional[str]
        落盘一次调用。
    end_turn(turn_id)
        结束一个 turn（清理 turn 计数器）。

    turn_id 通过 contextvars 传递，跨 async 无需显式透传。
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── 环境 ──

def _is_enabled() -> bool:
    return os.getenv("LLM_DEBUG_DUMP", "1") == "1"


def _base_dir() -> Path:
    return Path(os.getenv("LLM_DEBUG_DIR", "./logs/llm"))


def _max_chars() -> int:
    try:
        return int(os.getenv("LLM_DEBUG_MAX_CHARS", "100000"))
    except ValueError:
        return 100000


def _truncate(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    limit = _max_chars()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... [truncated at {limit} chars, total {len(text)}]"


# ── 当前 turn 的 context ──

_current_turn: contextvars.ContextVar[Optional["TurnContext"]] = (
    contextvars.ContextVar("llm_debug_turn", default=None)
)


@dataclass
class TurnContext:
    session_id: str
    task_id: str
    turn_id: str
    turn_index: int
    question: str
    turn_dir: Path
    counter: int = 0
    started_at: float = field(default_factory=time.time)

    def next_index(self) -> int:
        self.counter += 1
        return self.counter


def _slugify(text: str, max_len: int = 32) -> str:
    """把问题转成文件系统友好的短 slug。"""
    if not text:
        return "empty"
    # 去换行、去特殊字符、去重空格
    s = text.strip().replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    # 保留中英文、数字、常见符号，其余替换为 -
    s = re.sub(r"[^\w\u4e00-\u9fff\-]+", "-", s, flags=re.UNICODE)
    s = s.strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "empty"


# ── 会话级 turn 计数 ──
# key: (session_id, task_id) → 已开启的 turn 数
_turn_counters: Dict[tuple, int] = {}


def begin_turn(
    *,
    session_id: Optional[str],
    task_id: Optional[str],
    question: str,
) -> Optional[TurnContext]:
    """开启一个新 turn。

    调用位置：stream_task 收到用户提问时。返回 TurnContext（未启用时 None）。
    后续所有 dump_llm_call 都会落到该 turn 的目录。
    """
    if not _is_enabled():
        return None

    try:
        sid = session_id or "no-session"
        tid = task_id or "no-task"
        key = (sid, tid)
        idx = _turn_counters.get(key, 0) + 1
        _turn_counters[key] = idx

        turn_id = f"turn-{idx:02d}"
        slug = _slugify(question)
        dir_name = f"{turn_id}-ask-{slug}"

        turn_dir = _base_dir() / sid / tid / dir_name
        turn_dir.mkdir(parents=True, exist_ok=True)

        ctx = TurnContext(
            session_id=sid,
            task_id=tid,
            turn_id=turn_id,
            turn_index=idx,
            question=question,
            turn_dir=turn_dir,
        )

        # 写一份 meta.json，记录这次提问的原始内容
        try:
            meta_path = turn_dir / "00-meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session_id": sid,
                        "task_id": tid,
                        "turn_id": turn_id,
                        "turn_index": idx,
                        "question": question,
                        "started_at": datetime.now().isoformat(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            logger.exception("[dsh][llm-debug] meta.json 写入失败")

        _current_turn.set(ctx)
        logger.info(
            "[dsh][llm-debug] turn started: %s / %s / %s",
            sid, tid, dir_name,
        )
        return ctx

    except Exception:
        logger.exception("[dsh][llm-debug] begin_turn failed")
        return None


def end_turn() -> None:
    """结束当前 turn（清空 contextvar）。"""
    _current_turn.set(None)


def current_turn() -> Optional[TurnContext]:
    return _current_turn.get()


def reset_turn_counters(session_id: Optional[str] = None) -> None:
    """重置计数（重启后端时可选调用；一般不用）。"""
    if session_id is None:
        _turn_counters.clear()
    else:
        for k in list(_turn_counters.keys()):
            if k[0] == session_id:
                del _turn_counters[k]


# ── 落盘 ──

_KIND_ORDER = {
    "meta": 0,
    "decide": 10,
    "fill_params": 20,
    "tool_call": 30,
    "stream_answer": 40,
    "stream_answer_response": 50,
}


def dump_llm_call(
    *,
    kind: str,
    prompt: str,
    response: Optional[str] = None,
    model: Optional[str] = None,
    duration_ms: Optional[float] = None,
    tool_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Optional[str]:
    """落盘一次 LLM 调用到当前 turn 目录。返回文件路径。

    未启用 / 没有当前 turn 时返回 None。
    """
    if not _is_enabled():
        return None

    ctx = _current_turn.get()
    if ctx is None:
        logger.warning(
            "[dsh][llm-debug] dump_llm_call(%s) 但无当前 turn，已忽略", kind
        )
        return None

    try:
        idx = ctx.next_index()
        kind_part = kind
        if tool_name:
            kind_part = f"{kind}-{tool_name}"
        filename = f"{idx:02d}-{kind_part}.json"
        path = ctx.turn_dir / filename

        now = datetime.now()
        payload: Dict[str, Any] = {
            "timestamp": now.isoformat(),
            "session_id": ctx.session_id,
            "task_id": ctx.task_id,
            "turn_id": ctx.turn_id,
            "turn_index": ctx.turn_index,
            "call_index": idx,
            "kind": kind,
            "model": model,
            "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
            "tool_name": tool_name,
            "prompt_len": len(prompt) if prompt else 0,
            "response_len": len(response) if response else 0,
            "prompt": _truncate(prompt),
            "response": _truncate(response),
            "error": error,
        }
        if extra:
            payload["extra"] = extra

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return str(path)

    except Exception:
        logger.exception("[dsh][llm-debug] dump failed: kind=%s", kind)
        return None