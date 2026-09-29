"""把每次 LLM 调用落盘，按「会话 → 任务 → 提问」三级归类。

目录结构：
    logs/llm/
      <session_id>/
        <task_id>/
          turn-01-ask-<slug>/
            00-meta.json                 # turn 元数据
            all-calls.jsonl              # 所有调用，一行一条，追加写
            all-calls.json               # 所有调用，汇总 JSON，end_turn 时重建
            01-decide.json               # 单次调用详情
            02-fill_params-xxx.json
            ...

环境变量：
    LLM_DEBUG_DUMP=1                开启（默认 1）
    LLM_DEBUG_DIR=./logs/llm        自定义根目录
    LLM_DEBUG_MAX_CHARS=100000      单字段最大字符（超出截断）

关键 API：
    begin_turn(session_id, task_id, question) -> TurnContext
        开启一个 turn，后续所有 dump_llm_call 都归到该目录。
    dump_llm_call(kind=..., prompt=..., response=...) -> Optional[str]
        落盘一次调用（同时写单文件 all-calls.jsonl 与分文件）。
    end_turn()
        结束一个 turn（重建 all-calls.json、清理计数器）。

    turn_id 通过 contextvars 传递，跨 async 无需显式透传。
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import threading
import time
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


# ── 步骤元信息映射 ──

# kind → (中文标签, 分组)
_STEP_META: Dict[str, tuple] = {
    "meta":                   ("0. 元信息",          "meta"),
    "decide":                 ("1. 编排决策",        "orchestrator"),
    "fill_params":            ("2. 填参数",          "orchestrator"),
    "tool_call":              ("3. 调用工具",        "tool"),
    "stream_answer":          ("4. 生成回答",        "answer"),
    "stream_answer_response": ("5. 回答完成",        "answer"),
}


def _step_label(kind: str, tool_name: Optional[str]) -> str:
    base, _ = _STEP_META.get(kind, (kind, "other"))
    if tool_name:
        return f"{base} {tool_name}"
    return base


def _step_group(kind: str) -> str:
    _, group = _STEP_META.get(kind, (kind, "other"))
    return group


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
    # 单文件相关
    jsonl_path: Path = None
    json_path: Path = None
    dirty: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def next_index(self) -> int:
        self.counter += 1
        return self.counter


def _slugify(text: str, max_len: int = 32) -> str:
    if not text:
        return "empty"
    s = text.strip().replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\u4e00-\u9fff\-]+", "-", s, flags=re.UNICODE)
    s = s.strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "empty"


# ── 会话级 turn 计数 ──

_turn_counters: Dict[tuple, int] = {}
_counters_lock = threading.Lock()


def begin_turn(
    *,
    session_id: Optional[str],
    task_id: Optional[str],
    question: str,
) -> Optional[TurnContext]:
    """开启一个新 turn。"""
    if not _is_enabled():
        return None

    try:
        sid = session_id or "no-session"
        tid = task_id or "no-task"
        key = (sid, tid)
        with _counters_lock:
            idx = _turn_counters.get(key, 0) + 1
            _turn_counters[key] = idx

        turn_id = f"turn-{idx:02d}"
        slug = _slugify(question)
        dir_name = f"{turn_id}-ask-{slug}"

        turn_dir = _base_dir() / sid / tid / dir_name
        turn_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = turn_dir / "all-calls.jsonl"
        json_path = turn_dir / "all-calls.json"

        ctx = TurnContext(
            session_id=sid,
            task_id=tid,
            turn_id=turn_id,
            turn_index=idx,
            question=question,
            turn_dir=turn_dir,
            jsonl_path=jsonl_path,
            json_path=json_path,
        )

        # 清空 jsonl（同一目录若已存在，截断）
        try:
            jsonl_path.write_text("", encoding="utf-8")
        except Exception:
            logger.exception("[dsh][llm-debug] 清空 all-calls.jsonl 失败")

        # 写 meta.json
        try:
            meta = {
                "session_id": sid,
                "task_id": tid,
                "turn_id": turn_id,
                "turn_index": idx,
                "question": question,
                "started_at": datetime.now().isoformat(),
                "steps": [],
            }
            (turn_dir / "00-meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
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
    """结束当前 turn：重建 all-calls.json，清空 contextvar。"""
    ctx = _current_turn.get()
    if ctx is not None:
        try:
            _rebuild_aggregate(ctx)
        except Exception:
            logger.exception("[dsh][llm-debug] 重建 all-calls.json 失败")
    _current_turn.set(None)


def current_turn() -> Optional[TurnContext]:
    return _current_turn.get()


def reset_turn_counters(session_id: Optional[str] = None) -> None:
    with _counters_lock:
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


def _build_call_record(
    *,
    ctx: TurnContext,
    idx: int,
    kind: str,
    prompt: str,
    response: Optional[str],
    model: Optional[str],
    duration_ms: Optional[float],
    tool_name: Optional[str],
    extra: Optional[Dict[str, Any]],
    error: Optional[str],
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "session_id": ctx.session_id,
        "task_id": ctx.task_id,
        "turn_id": ctx.turn_id,
        "turn_index": ctx.turn_index,

        # 步骤信息
        "step_index": idx,
        "step_label": _step_label(kind, tool_name),
        "step_group": _step_group(kind),

        "kind": kind,
        "tool_name": tool_name,
        "model": model,
        "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
        "prompt_len": len(prompt) if prompt else 0,
        "response_len": len(response) if response else 0,
        "prompt": _truncate(prompt),
        "response": _truncate(response),
        "error": error,
        "extra": extra or None,
    }


def _append_jsonl(ctx: TurnContext, record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with ctx.lock:
        with open(ctx.jsonl_path, "a", encoding="utf-8") as f:
            f.write(line)
    ctx.dirty = True


def _rebuild_aggregate(ctx: TurnContext) -> None:
    """从 jsonl 重建 all-calls.json。"""
    if not ctx.jsonl_path.exists():
        return

    records: List[Dict[str, Any]] = []
    with open(ctx.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "[dsh][llm-debug] jsonl 行解析失败，已跳过: %r", line[:120]
                )

    aggregate = {
        "session_id": ctx.session_id,
        "task_id": ctx.task_id,
        "turn_id": ctx.turn_id,
        "turn_index": ctx.turn_index,
        "question": ctx.question,
        "started_at": datetime.fromtimestamp(ctx.started_at).isoformat(),
        "finished_at": datetime.now().isoformat(),
        "total_calls": len(records),
        "steps": [
            {
                "step_index": r["step_index"],
                "step_label": r["step_label"],
                "step_group": r["step_group"],
                "kind": r["kind"],
                "tool_name": r.get("tool_name"),
                "duration_ms": r.get("duration_ms"),
                "error": r.get("error"),
                "timestamp": r.get("timestamp"),
            }
            for r in records
        ],
        "calls": records,
    }

    ctx.json_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ctx.dirty = False


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
    """落盘一次 LLM 调用。

    同时写：
        - 分文件：<NN>-<kind>[-<tool>].json
        - 单文件追加：all-calls.jsonl
        - 单文件汇总：all-calls.json（仅在 end_turn 时重建）
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

        # 1. 分文件
        kind_part = kind
        if tool_name:
            kind_part = f"{kind}-{tool_name}"
        filename = f"{idx:02d}-{kind_part}.json"
        path = ctx.turn_dir / filename

        record = _build_call_record(
            ctx=ctx,
            idx=idx,
            kind=kind,
            prompt=prompt,
            response=response,
            model=model,
            duration_ms=duration_ms,
            tool_name=tool_name,
            extra=extra,
            error=error,
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        # 2. 单文件追加
        _append_jsonl(ctx, record)

        return str(path)

    except Exception:
        logger.exception("[dsh][llm-debug] dump failed: kind=%s", kind)
        return None


# ── 手动触发聚合重建（如需要中间快照） ──

def flush_aggregate() -> None:
    """把当前 turn 的 jsonl 重建为 all-calls.json。"""
    ctx = _current_turn.get()
    if ctx is None:
        return
    try:
        _rebuild_aggregate(ctx)
    except Exception:
        logger.exception("[dsh][llm-debug] flush_aggregate 失败")