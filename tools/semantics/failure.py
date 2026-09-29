"""Tool failure classification + recovery hints.

dsh 风格：**默认信任 LLM 能读原文**，只在极少数情况补充
LLM 从原文拿不到的信息。

只保留一类 hint：
    - TOOL_NOT_FOUND：LLM 不知道"当前有哪些相似工具"——原文里没有

其他所有错误一律返回空 hint：
    - 参数名错：`execute_tool_async` 拼的原文已含
               "工具接受的关键字 + 你传入的"
    - 参数值错：服务器原文里已含修正建议
    - 权限/超时/限流/404：LLM 自己读原文能判断
"""

from __future__ import annotations

from typing import Dict, List, Optional


def classify_failure(error_message: str) -> str:
    """只分类「LLM 从原文拿不到信息」的情况。

    其余一律 UNCLASSIFIED → 上层不输出 hint。
    """
    err = (error_message or "").lower()

    if any(kw in err for kw in (
        "tool not found", "unknown tool", "no such tool",
        "tool_not_found",
    )):
        return "TOOL_NOT_FOUND"

    # 其他所有情况：不分类。让 LLM 读原文。
    return "UNCLASSIFIED"


def _find_similar_tools(
    failed_tool: str,
    available_tools: Optional[List[str]],
) -> List[str]:
    """找同类工具。

    优先同 server（如 mcp__filesystem__write_file → 其他 mcp__filesystem__*），
    再退到全局关键词匹配。
    """
    if not available_tools:
        return []

    basename = failed_tool.rsplit("__", 1)[-1].lower()

    # 同 server 前缀
    server_prefix = ""
    if failed_tool.startswith("mcp__"):
        parts = failed_tool.split("__", 2)
        if len(parts) == 3:
            server_prefix = f"mcp__{parts[1]}__"

    # 功能关键词分组
    if any(k in basename for k in ("write", "edit", "create", "save")):
        kws = ("write", "edit", "create", "save")
    elif any(k in basename for k in ("read", "get", "cat", "open")):
        kws = ("read", "get", "cat", "open")
    elif any(k in basename for k in ("move", "rename")):
        kws = ("move", "rename")
    elif any(k in basename for k in ("search", "find", "grep", "query")):
        kws = ("search", "find", "grep", "query")
    elif any(k in basename for k in ("list", "dir", "tree")):
        kws = ("list", "dir", "tree")
    elif any(k in basename for k in ("delete", "remove")):
        kws = ("delete", "remove")
    else:
        # 无关键词：只返回同 server 的其他工具
        if server_prefix:
            return [
                name for name in available_tools
                if name.startswith(server_prefix) and name != failed_tool
            ][:5]
        return []

    # 同 server 优先
    same_server = [
        name for name in available_tools
        if server_prefix
        and name.startswith(server_prefix)
        and name != failed_tool
        and any(kw in name.lower() for kw in kws)
    ]
    if same_server:
        return same_server[:5]

    # 退到全局匹配
    return [
        name for name in available_tools
        if name != failed_tool
        and any(kw in name.lower() for kw in kws)
    ][:5]


def get_failure_recovery_hint(
    failure_type: str,
    error_message: str,
    failed_tool_name: str,
    available_tools: Optional[List[str]] = None,
) -> str:
    """返回补充信息块——**只补 LLM 从原文拿不到的信息**。

    其他情况返回空字符串，让 LLM 读原文。
    """
    if failure_type == "TOOL_NOT_FOUND":
        similar = _find_similar_tools(failed_tool_name, available_tools)
        if not similar:
            return ""
        lines = ["当前可用的相似工具："]
        for name in similar[:5]:
            lines.append(f"  - {name}")
        return "\n".join(lines) + "\n"

    return ""