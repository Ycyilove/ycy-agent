"""answer prompt 组装：意图注入 + build_prompt + history 裁剪。"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from core.llm.prompts import build_prompt, attachment_label

logger = logging.getLogger(__name__)


def inject_memory_intent(
    session_id: str,
    tool_context: Optional[List[str]],
) -> List[str]:
    """把 mem0 里 user_intent 类事实注入 tool_context。失败静默。"""
    try:
        from core.memory.mem0_layer import recall_recent_facts
        _facts = recall_recent_facts(session_id, limit=6) if session_id else []
        _intent_facts = [
            f for f in (_facts or [])
            if f and not re.match(r"^[A-Za-z]:[\\/]|^/", f)
        ][:3]
        if not _intent_facts:
            return list(tool_context or [])
        _lines = ["## 本会话已知意图（长期记忆）"]
        for f in _intent_facts:
            _lines.append(f"- {f[:150]}")
        _lines.append(
            "\n**注意**：这些是历史意图，可能已过时；"
            "与当前问题无关时忽略。"
        )
        logger.info(
            "[dsh][answer] injected %d intent facts", len(_intent_facts),
        )
        return list(tool_context or []) + ["\n".join(_lines)]
    except Exception:
        logger.exception("recall_recent_facts (answer) failed")
        return list(tool_context or [])


def trim_history(
    msgs: List[Dict[str, str]],
    max_user_chars: int = 800,
    max_assistant_chars: int = 400,
) -> List[Dict[str, str]]:
    out = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content") or ""
        limit = max_user_chars if role == "user" else max_assistant_chars
        if len(content) > limit:
            content = content[:limit] + "..."
        out.append({"role": role, "content": content})
    return out


def build_answer_messages(
    *,
    message: str,
    mode: str,
    attachments: List[Dict[str, Any]],
    rag_context: Optional[str],
    tool_context: Optional[List[str]],
    prior_messages: List[Dict[str, str]],
    task_id: str,
    session_provider,
) -> List[Dict[str, str]]:
    """组装最终 messages。包含终止原因 block。"""
    stop_block = ""
    if task_id:
        try:
            task = session_provider().get_task(task_id)
            stop_reason = task.get("stop_reason") if task else None
            stop_detail = task.get("stop_detail") if task else None
        except Exception:
            stop_reason, stop_detail = None, None
        if stop_reason and stop_reason != "done":
            stop_block = (
                f"\n\n【终止原因】{stop_reason}"
                + (f"：{stop_detail}" if stop_detail else "")
                + "\n请在回答里如实说明未完成的部分与原因，"
                "不要声称任务已完全完成。"
            )

    prompt = build_prompt(
        message,
        mode,
        attachments,
        rag_context,
        tool_context=tool_context,
        extra_suffix=stop_block,
    )
    return trim_history(prior_messages) + [{"role": "user", "content": prompt}]


__all__ = [
    "attachment_label",
    "inject_memory_intent",
    "trim_history",
    "build_answer_messages",
]