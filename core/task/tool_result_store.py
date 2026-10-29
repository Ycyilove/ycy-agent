"""工具结果卸载存储：大结果落盘 + 句柄化。

设计参考：
    - LangChain Deep Agents: >20K tokens 写文件，上下文留路径 + 前 10 行
    - Claude Code: >50K 字符自动落盘，对话中替换为文件引用
    - AgentScope: 截断时追加 <<<TRUNCATED>>> + 文件路径提示
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULT_DIR = os.path.join(_PROJECT_ROOT, "data", "tool_results")


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文 ~1.5 字符/token，英文 ~4 字符/token）。"""
    if not text:
        return 0
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cn
    return int(cn / 1.5 + other / 4)


def _safe_task_dir(task_id: str) -> str:
    """把 task_id 转成安全目录名（防路径穿越）。"""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", task_id or "_default")[:64]
    return safe or "_default"


def _ensure_dir(task_id: str) -> str:
    d = os.path.join(_RESULT_DIR, _safe_task_dir(task_id))
    os.makedirs(d, exist_ok=True)
    return d


def save_tool_result(
    task_id: str,
    step_index: int,
    tool_name: str,
    payload: Any,
) -> str:
    """完整结果落盘，返回 result_id（同时是文件名 stem）。"""
    d = _ensure_dir(task_id)
    result_id = f"tr-{step_index:02d}-{uuid.uuid4().hex[:8]}"
    safe_tool = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)[:40]
    path = os.path.join(d, f"{result_id}-{safe_tool}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        logger.exception("[tool_result_store] 落盘失败: %s", path)
        return ""
    return result_id


def _find_exact(d: str, result_id: str) -> Optional[str]:
    """精确匹配 <result_id>.json 或 <result_id>-*.json（前缀 + 分隔符）。"""
    if not result_id:
        return None
    prefix = result_id + "-"
    exact = os.path.join(d, f"{result_id}.json")
    if os.path.isfile(exact):
        return exact
    for fn in os.listdir(d):
        if fn.endswith(".json") and (
            fn == f"{result_id}.json" or fn.startswith(prefix)
        ):
            return os.path.join(d, fn)
    return None


def load_tool_result(task_id: str, result_id: str) -> Optional[Dict[str, Any]]:
    d = os.path.join(_RESULT_DIR, _safe_task_dir(task_id))
    if not os.path.isdir(d):
        return None
    path = _find_exact(d, result_id)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("[tool_result_store] 读取失败: %s", path)
        return None


def find_result_path(task_id: str, result_id: str) -> Optional[str]:
    d = os.path.join(_RESULT_DIR, _safe_task_dir(task_id))
    if not os.path.isdir(d):
        return None
    return _find_exact(d, result_id)


def summarize_search_knowledge(
    payload: Dict[str, Any],
    per_file_chunks: int = 2,
    chunk_preview_chars: int = 400,
) -> Dict[str, Any]:
    """对 search_knowledge 的 chunks 按文档采样，生成摘要。"""
    chunks = payload.get("chunks") or []
    if not isinstance(chunks, list):
        return {"files": [], "total_chunks": 0}

    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        if not isinstance(c, dict):
            continue
        fn = c.get("filename") or "_unknown"
        by_file.setdefault(fn, []).append(c)

    files_summary = []
    for fn, items in by_file.items():
        previews = []
        for c in items[:per_file_chunks]:
            text = (c.get("text") or "").strip()
            previews.append({
                "chunk_index": items.index(c),
                "preview": text[:chunk_preview_chars],
                "resources": [
                    {"id": r.get("id"), "url": r.get("url"),
                     "filename": r.get("filename"), "page_no": r.get("page_no")}
                    for r in (c.get("resources") or [])[:2]
                ],
            })
        files_summary.append({
            "filename": fn,
            "chunk_count_in_result": len(items),
            "previews": previews,
        })

    return {
        "query": payload.get("query"),
        "mode": payload.get("mode"),
        "total_chunks": len(chunks),
        "files": files_summary,
    }


def build_handle_summary(
    task_id: str,
    step_index: int,
    tool_name: str,
    payload: Any,
    *,
    max_inline_tokens: int = 4000,
) -> Dict[str, Any]:
    """决定是"内联"还是"落盘 + 摘要"，返回给 LLM 的 dict。

    返回结构：
        {
          "inline": true/false,
          "result_id": "tr-xx" | None,
          "handle_path": "data/tool_results/..." | None,
          "summary": {...} | None,
          "payload": {...} | None,     # inline=true 时给完整
        }
    """
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = str(payload)

    tokens = _estimate_tokens(text)
    if tokens <= max_inline_tokens:
        return {"inline": True, "payload": payload}

    # 大结果：落盘
    result_id = save_tool_result(task_id, step_index, tool_name, payload)
    handle_path = None
    if result_id:
        p = find_result_path(task_id, result_id)
        if p:
            handle_path = os.path.relpath(p, _PROJECT_ROOT).replace("\\", "/")

    # 针对 search_knowledge 生成按文档采样的摘要
    summary = None
    if tool_name == "search_knowledge" and isinstance(payload, dict):
        # 动态决定 per_file_chunks：文件越少，每文件给越多 chunk
        chunks = payload.get("chunks") or []
        file_set = {
            c.get("filename") for c in chunks
            if isinstance(c, dict) and c.get("filename")
        }
        file_count = max(1, len(file_set))
        # 目标：摘要总 token 不超过 max_inline_tokens 的一半
        target_chunks = max(4, (max_inline_tokens // 2) // 400)  # 400 字/chunk
        per_file = max(1, target_chunks // file_count)
        summary = summarize_search_knowledge(
            payload, per_file_chunks=per_file, chunk_preview_chars=400
        )
        logger.info(
            "[tool_result_store] search_knowledge 摘要: %d 文件, 每文件 %d chunk",
            file_count, per_file,
        )

    return {
        "inline": False,
        "result_id": result_id,
        "handle_path": handle_path,
        "summary": summary,
        "estimated_tokens": tokens,
    }