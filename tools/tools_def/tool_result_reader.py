"""按需读取卸载的工具结果。

大工具结果（如 search_knowledge 返回 15 个 chunk）会被
core/task/tool_result_store 卸载到 data/tool_results/<task_id>/，
LLM 通过本工具按 filename / start_index 分页读取完整内容。

注册方式：不用 @tool 装饰器（它不透传 idempotent），
直接调 registry.register_callable，确保 idempotent=True。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from core.task.tool_result_store import load_tool_result
from .. import get_registry

logger = logging.getLogger(__name__)

READ_TOOL_RESULT_PARAMETERS: Dict[str, Any] = {
    "result_id": {
        "type": "string",
        "description": "工具返回的 result_id（如 tr-03-a1b2c3d4）",
    },
    "task_id": {
        "type": "string",
        "description": "当前任务 ID（从 SSE task 事件或上下文获取）",
    },
    "filename": {
        "type": "string",
        "description": "只读某个文档的 chunk（可选，search_knowledge 场景用）",
    },
    "chunk_index": {
        "type": "integer",
        "description": "chunk 场景：从第几个 chunk 开始（从 0 开始），默认 0",
    },
    "chunk_count": {
        "type": "integer",
        "description": "chunk 场景：最多返回几个 chunk，默认 3",
    },
    "char_offset": {
        "type": "integer",
        "description": "通用场景：字符偏移，默认 0",
    },
    "char_limit": {
        "type": "integer",
        "description": "通用场景：最多返回多少字符，默认 4000",
    },
}


def read_tool_result(
    result_id: str,
    task_id: str,
    filename: Optional[str] = None,
    chunk_index: int = 0,
    chunk_count: int = 3,
    char_offset: int = 0,
    char_limit: int = 4000,
    **legacy,
) -> Dict[str, Any]:
    # 兼容旧参数 start_index / limit
    if "start_index" in legacy:
        chunk_index = int(legacy["start_index"])
        char_offset = int(legacy["start_index"])
    if "limit" in legacy:
        chunk_count = int(legacy["limit"])
        char_limit = int(legacy["limit"]) * 2000
    """读取之前工具返回的大结果的完整内容。"""
    payload = load_tool_result(task_id, result_id)
    if payload is None:
        return {
            "success": False,
            "error": (
                f"result_id {result_id} 在 task {task_id} 下不存在；"
                f"请确认 task_id 和 result_id 是否正确"
            ),
        }

    # search_knowledge 风格：按 filename + start_index 切片
    chunks = payload.get("chunks") if isinstance(payload, dict) else None
    if isinstance(chunks, list):
        if filename:
            chunks = [
                c for c in chunks
                if isinstance(c, dict) and c.get("filename") == filename
            ]
        sliced = chunks[chunk_index: chunk_index + max(1, chunk_count)]
        return {
            "success": True,
            "result_id": result_id,
            "filename": filename,
            "chunk_index": chunk_index,
            "returned": len(sliced),
            "total": len(chunks),
            "chunks": sliced,
        }

    # 通用：JSON 文本分页
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = str(payload)
    start = max(0, char_offset)
    end = start + max(1, char_limit)
    return {
        "success": True,
        "result_id": result_id,
        "char_offset": start,
        "returned_chars": len(text[start:end]),
        "total_chars": len(text),
        "content": text[start:end],
    }


# ── 显式注册（不用 @tool，因为它不透传 idempotent） ──
get_registry().register_callable(
    read_tool_result,
    name="read_tool_result",
    description=(
        "读取之前工具返回的大结果的完整内容。"
        "当工具返回里有 `result_id` 或 `handle` 字段时，说明结果被卸载到磁盘，"
        "可用本工具按需读取。\n\n"
        "**使用场景**：\n"
        "- `search_knowledge` 返回的摘要只包含每个文档的部分 chunk，"
        "  想看某个文档的更多 chunk 时，用 "
        "`result_id` + `filename` + `chunk_index` + `chunk_count` 读取。\n"
        "- 想看完整 JSON 时，用 `char_offset` + `char_limit` 分页。\n"
        "- **chunk 场景和字符场景不要混用参数**：\n"
        "  * chunk 场景用 `chunk_index` / `chunk_count`\n"
        "  * 字符场景用 `char_offset` / `char_limit`\n"
    ),
    parameters=READ_TOOL_RESULT_PARAMETERS,
    required=["result_id", "task_id"],
    category="knowledge_read",
    danger_level="safe",
    idempotent=True,
)