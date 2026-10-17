"""把 Mem0 的写入/检索落盘，按「会话 → 任务」归类。

目录结构：
    logs/mem0/
      <session_id>/
        <task_id>/
          events.jsonl                 # 所有事件，一行一条，追加写
          events.json                  # 汇总，flush 时重建
          summary.json                 # 该任务的事实列表快照

环境变量：
    MEM0_DEBUG_DUMP=1               开启（默认 1）
    MEM0_DEBUG_DIR=./logs/mem0      自定义根目录
    MEM0_DEBUG_MAX_CHARS=50000      单字段最大字符

关键 API：
    begin_task(session_id, task_id) -> TaskContext
    record_add(session_id, fact, fact_type, success, error=None)
    record_recall(session_id, limit, results)
    record_skip(reason, tool_name, params)
    end_task()
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── 环境 ──

def _is_enabled() -> bool:
    return os.getenv("MEM0_DEBUG_DUMP", "1") == "1"


def _base_dir() -> Path:
    return Path(os.getenv("MEM0_DEBUG_DIR", "./logs/mem0"))


def _max_chars() -> int:
    try:
        return int(os.getenv("MEM0_DEBUG_MAX_CHARS", "50000"))
    except ValueError:
        return 50000


def _truncate(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    limit = _max_chars()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, total {len(text)}]"


# ── 当前 task 上下文 ──

_current_task: contextvars.ContextVar[Optional["TaskContext"]] = (
    contextvars.ContextVar("mem0_debug_task", default=None)
)


@dataclass
class TaskContext:
    session_id: str
    task_id: str
    task_dir: Path
    jsonl_path: Path
    json_path: Path
    summary_path: Path
    started_at: float = field(default_factory=time.time)
    counter: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    # 事实快照：{fact_type: [fact, ...]}
    facts: Dict[str, List[str]] = field(default_factory=dict)
    # 统计
    stats: Dict[str, int] = field(default_factory=lambda: {
        "add_ok": 0, "add_fail": 0,
        "recall_ok": 0, "recall_fail": 0, "recall_empty": 0,
        "skip": 0,
    })

    def next_index(self) -> int:
        self.counter += 1
        return self.counter


# ── API ──

def begin_task(
    *,
    session_id: Optional[str],
    task_id: Optional[str],
) -> Optional[TaskContext]:
    """开启一个 task 的 mem0 记录。"""
    if not _is_enabled():
        return None

    try:
        sid = session_id or "no-session"
        tid = task_id or "no-task"

        task_dir = _base_dir() / sid / tid
        task_dir.mkdir(parents=True, exist_ok=True)

        ctx = TaskContext(
            session_id=sid,
            task_id=tid,
            task_dir=task_dir,
            jsonl_path=task_dir / "events.jsonl",
            json_path=task_dir / "events.json",
            summary_path=task_dir / "summary.json",
        )

        # 清空 jsonl（同一目录复用）
        try:
            ctx.jsonl_path.write_text("", encoding="utf-8")
        except Exception:
            logger.exception("[dsh][mem0-debug] 清空 events.jsonl 失败")

        _current_task.set(ctx)
        logger.info(
            "[dsh][mem0-debug] task started: %s / %s", sid, tid,
        )
        return ctx

    except Exception:
        logger.exception("[dsh][mem0-debug] begin_task failed")
        return None


def end_task() -> None:
    """结束当前 task：重建 events.json + 写 summary.json。"""
    ctx = _current_task.get()
    if ctx is None:
        return
    try:
        _rebuild_aggregate(ctx)
        _write_summary(ctx)
    except Exception:
        logger.exception("[dsh][mem0-debug] end_task 失败")
    _current_task.set(None)


def current_task() -> Optional[TaskContext]:
    return _current_task.get()


# ── 写入 / 检索 记录 ──

def record_add(
    *,
    session_id: str,
    fact: str,
    fact_type: str,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """记录一次 Mem0 写入尝试。"""
    ctx = _current_task.get()
    if ctx is None or not _is_enabled():
        return

    try:
        idx = ctx.next_index()
        record = {
            "timestamp": datetime.now().isoformat(),
            "index": idx,
            "event": "add",
            "session_id": session_id,
            "fact": _truncate(fact),
            "fact_type": fact_type,
            "success": success,
            "error": error,
        }

        if success:
            ctx.stats["add_ok"] += 1
            ctx.facts.setdefault(fact_type, []).append(fact)
        else:
            ctx.stats["add_fail"] += 1

        _append(ctx, record)
    except Exception:
        logger.exception("[dsh][mem0-debug] record_add 失败")


def record_recall(
    *,
    session_id: str,
    limit: int,
    results: List[str],
) -> None:
    """记录一次 Mem0 检索。"""
    ctx = _current_task.get()
    if ctx is None or not _is_enabled():
        return

    try:
        idx = ctx.next_index()
        record = {
            "timestamp": datetime.now().isoformat(),
            "index": idx,
            "event": "recall",
            "session_id": session_id,
            "limit": limit,
            "hit_count": len(results),
            "results": results,
        }

        if len(results) == 0:
            ctx.stats["recall_empty"] += 1
        else:
            ctx.stats["recall_ok"] += 1

        _append(ctx, record)
    except Exception:
        logger.exception("[dsh][mem0-debug] record_recall 失败")


def record_skip(
    *,
    reason: str,
    tool_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """记录一次「本该写但跳过」的事件（便于判断逻辑是否正确）。"""
    ctx = _current_task.get()
    if ctx is None or not _is_enabled():
        return

    try:
        idx = ctx.next_index()
        record = {
            "timestamp": datetime.now().isoformat(),
            "index": idx,
            "event": "skip",
            "reason": reason,
            "tool_name": tool_name,
            "params": params,
        }
        ctx.stats["skip"] += 1
        _append(ctx, record)
    except Exception:
        logger.exception("[dsh][mem0-debug] record_skip 失败")


# ── 内部落盘 ──

def _append(ctx: TaskContext, record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with ctx.lock:
        with open(ctx.jsonl_path, "a", encoding="utf-8") as f:
            f.write(line)


def _rebuild_aggregate(ctx: TaskContext) -> None:
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
                    "[dsh][mem0-debug] 行解析失败: %r", line[:120],
                )

    aggregate = {
        "session_id": ctx.session_id,
        "task_id": ctx.task_id,
        "started_at": datetime.fromtimestamp(ctx.started_at).isoformat(),
        "finished_at": datetime.now().isoformat(),
        "total_events": len(records),
        "stats": dict(ctx.stats),
        "events": records,
    }

    ctx.json_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_summary(ctx: TaskContext) -> None:
    """写一份「该任务写入了哪些事实」的快照。"""
    summary = {
        "session_id": ctx.session_id,
        "task_id": ctx.task_id,
        "generated_at": datetime.now().isoformat(),
        "stats": dict(ctx.stats),
        "facts": ctx.facts,
    }
    ctx.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )