"""工具 schema 统一读取工具。

供 AgentOrchestrator（_fill_params）和 ToolAgent（execute_tool_async）
共用，避免口径分叉。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def read_tool_schema(meta: Any) -> Tuple[Dict[str, Any], List[str]]:
    """统一读取工具 schema，兼容 input_schema / parameters 两种存放方式。

    Returns:
        (properties, required)
        - 内置工具（修复 B 后）：input_schema 有值
        - MCP 工具：input_schema 或 parameters
        - 老工具：parameters 兜底
    """
    schema = getattr(meta, "input_schema", None) or {}

    props = schema.get("properties") or {}
    if not props:
        raw = getattr(meta, "parameters", None) or {}
        if isinstance(raw, dict) and "properties" in raw:
            props = raw.get("properties") or {}
        elif isinstance(raw, dict):
            props = raw
        else:
            props = {}

    required = list(schema.get("required") or [])
    if not required:
        raw = getattr(meta, "parameters", None) or {}
        if isinstance(raw, dict):
            required = list(raw.get("required") or [])
    if not required:
        required = list(getattr(meta, "required", None) or [])

    return props, required