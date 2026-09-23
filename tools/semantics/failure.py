"""Tool failure classification + recovery hints.

策略 C：把失败分类，每类给出不同的恢复提示。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def classify_failure(error_message: str) -> str:
    """根据错误文本分类。"""
    err = (error_message or "").lower()

    if any(kw in err for kw in (
        "tool not found", "unknown tool", "does not exist",
        "no such tool", "tool_not_found",
    )):
        return "TOOL_NOT_FOUND"

    if any(kw in err for kw in (
        "missing required argument", "unexpected keyword argument",
        "missing required '", "unexpected '",
        "missing required \"", "unexpected \"",
    )):
        return "PARAM_NAME_ERROR"

    if any(kw in err for kw in (
        "invalid value", "couldn't find", "could not find",
        "out of range", "not supported", "valid values",
        "should be", "use format", "try the nearest",
    )):
        return "PARAM_VALUE_ERROR"

    if any(kw in err for kw in (
        "permission denied", "forbidden", "unauthorized",
        "access denied",
    )):
        return "PERMISSION_ERROR"

    if any(kw in err for kw in ("rate limit", "too many requests")):
        return "RATE_LIMIT"

    if any(kw in err for kw in ("timeout", "timed out")):
        return "TIMEOUT"

    if any(kw in err for kw in (
        "no agents found", "no results found", "not found",
        "该能力不存在", "未找到匹配",
    )):
        return "BUSINESS_TERMINAL"

    return "RECOVERABLE_GENERIC"


def _extract_param_names(error_message: str) -> Dict[str, List[str]]:
    """从参数错误里提取 missing / unexpected 参数名。"""
    import re
    msg = error_message or ""

    missing = re.findall(
        r"([A-Za-z_][A-Za-z0-9_]*)\s+Missing required argument", msg,
    )
    unexpected = re.findall(
        r"([A-Za-z_][A-Za-z0-9_]*)\s+Unexpected keyword argument", msg,
    )
    missing += re.findall(
        r"missing required ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", msg,
    )
    unexpected += re.findall(
        r"unexpected ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", msg,
    )
    return {
        "missing": sorted(set(missing)),
        "unexpected": sorted(set(unexpected)),
    }


def _find_similar_tools(
    failed_tool: str,
    available_tools: Optional[List[str]],
) -> List[str]:
    """根据失败工具的功能，在可用工具里找同类。"""
    if not available_tools:
        return []

    basename = failed_tool.rsplit("__", 1)[-1].lower()

    if "write" in basename or "edit" in basename:
        kws = ("write", "edit", "create")
    elif "read" in basename or "get" in basename:
        kws = ("read", "get")
    elif "move" in basename or "rename" in basename:
        kws = ("move", "rename")
    elif "search" in basename or "find" in basename:
        kws = ("search", "find")
    elif "dir" in basename or "list" in basename:
        kws = ("list", "dir")
    else:
        return []

    return [
        name for name in available_tools
        if any(kw in name.lower() for kw in kws)
    ]


def get_failure_recovery_hint(
    failure_type: str,
    error_message: str,
    failed_tool_name: str,
    available_tools: Optional[List[str]] = None,
) -> str:
    """返回格式化的恢复提示块，用于置顶到工具返回结果。"""
    if failure_type == "TOOL_NOT_FOUND":
        similar = _find_similar_tools(failed_tool_name, available_tools)
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "🔧 工具名不存在（**不是服务器问题**）",
            f"   你调用了 `{failed_tool_name}`，但该工具未注册。",
        ]
        if similar:
            lines.append("   **可用相似工具**：")
            for name in similar[:5]:
                lines.append(f"     - {name}")
            lines.append("   **下一步：用上述工具重试，不要用同一个错误名。**")
        else:
            lines.append("   **下一步：查看可用工具列表，找功能相同的工具。**")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines) + "\n\n"

    if failure_type == "PARAM_NAME_ERROR":
        params = _extract_param_names(error_message)
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "🔧 参数名错误（**不是服务器能力问题**）",
        ]
        if params["missing"]:
            lines.append(f"   你必须传以下参数（当前没传）：{params['missing']}")
        if params["unexpected"]:
            lines.append(f"   以下参数你多传了（必须删掉）：{params['unexpected']}")
        lines.append("   **不要换服务器！** 修正参数名后重试**同一个工具**。")
        lines.append("   **下一步立即用正确的参数名重试**。")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines) + "\n\n"

    if failure_type == "PARAM_VALUE_ERROR":
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "🔧 参数值错误（**服务器给了修正建议**）",
            "   **不要换服务器！也不要重新搜索！**",
            "   服务器的建议（你必须遵循）：",
        ]
        suggestion = (error_message or "")[:400]
        for line in suggestion.splitlines():
            lines.append(f"     {line}")
        lines.append("   **下一步：按上述建议修改参数值，重试同一个工具。**")
        lines.append(
            "   **如果传的是中文地名，先翻译成英文城市级名**"
            "（例如 \"广州天河区\" → \"Guangzhou\"），"
            "必要时加国家后缀（如 \"Guangzhou, China\"）。"
        )
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines) + "\n\n"

    if failure_type == "BUSINESS_TERMINAL":
        return (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⛔ 业务终态错误，**立即返回 done=true 并告知用户**。\n"
            "   不要尝试其他工具或重试。\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )

    if failure_type == "PERMISSION_ERROR":
        return (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🔒 权限错误。**不要重试同一个工具**。\n"
            "   尝试换一个工具，或返回 done=true 告知用户需要授权。\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )

    if failure_type == "TIMEOUT":
        return (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⏱️ 超时。**不要重试同一个工具**。\n"
            "   尝试换工具，或返回 done=true 告知用户。\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )

    if failure_type == "RATE_LIMIT":
        return (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🚦 频率限制。**稍后重试或换工具**。\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )

    return ""