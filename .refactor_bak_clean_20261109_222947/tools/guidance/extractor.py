"""MCP guidance extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


# 扫描用的引导字段名
_GUIDANCE_KEYS = (
    "next_action",
    "error",
    "message_detail",
    "hint",
    "suggestion",
    "guidance",
    "detail",
    "details",
    "reason",
)

# 认证相关关键词
_AUTH_KEYWORDS = (
    "missing", "required", "unauthenticated", "unauthorized",
    "not registered", "invalid", "expired", "forbidden",
    "api key", "api-key", "x-api-key", "bearer",
    "register", "onboard",
)

_AUTH_ERROR_KEYWORDS = _AUTH_KEYWORDS[:-2]  # 去掉 register / onboard

_PARAM_ERROR_KEYWORDS = (
    "invalid parameter", "missing parameter", "required parameter",
    "schema", "validation", "bad request", "unprocessable",
)

# 认证修复类工具关键词
_AUTH_FIX_TOOL_KEYWORDS = (
    "register", "onboard", "auth", "login", "sign",
    "create_key", "create_account", "get_key",
    "confirm_key", "confirm_keys", "persist",
    "verify_key", "activate",
)

# 参数修复类工具关键词
_PARAM_FIX_TOOL_KEYWORDS = (
    "schema", "describe", "list", "info", "get", "help",
)

_TOOL_NAME_PATTERNS = (
    re.compile(r"call\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?", re.IGNORECASE),
    re.compile(r"invoke\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?", re.IGNORECASE),
    re.compile(r"execute\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?", re.IGNORECASE),
    re.compile(r"(?:第一步|下一步|首先)[：:]\s*`?([a-zA-Z_][a-zA-Z0-9_]*)`?"),
)

_TOOL_NAME_FIELDS = ("tool", "next_tool", "tool_name", "name", "action")
_ARGS_FIELDS = ("arguments", "args", "parameters", "params")
_TOOL_NAME_STOPWORDS = frozenset(
    ("the", "a", "an", "this", "that", "none", "null", "true", "false")
)


def _first_text_block(content_blocks) -> str:
    if not isinstance(content_blocks, list):
        return ""
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text") or ""
            if text:
                return text
    return ""


def _classify_path(path: Tuple[str, ...]) -> Tuple[str, int, str]:
    """根据路径判断 (type, priority, source)。"""
    if not path:
        return ("optional", 20, "fallback")

    parts = [str(p).lower() for p in path]

    if "error" in parts and ("next" in parts or "next_action" in parts):
        return ("mandatory", 100, "error_next")
    if "next_action" in parts:
        return ("optional", 60, "next_action")
    if any(k in parts for k in ("guidance", "hint", "suggestion")):
        return ("optional", 50, "guidance_like")
    if any(k in parts for k in ("detail", "details", "reason")):
        return ("optional", 40, "detail_like")
    return ("optional", 30, "generic")


def _record_candidate(result: dict, tool_name: str, path: Tuple[str, ...]) -> None:
    if not isinstance(tool_name, str):
        return
    tool_name = tool_name.strip()
    if not tool_name or tool_name.lower() in _TOOL_NAME_STOPWORDS:
        return

    gtype, priority, source = _classify_path(path)
    candidates = result.setdefault("_candidates", [])
    for c in candidates:
        if c["tool"] == tool_name and c["path"] == tuple(path):
            return

    candidates.append({
        "tool": tool_name,
        "path": tuple(path),
        "type": gtype,
        "priority": priority,
        "source": source,
    })


def _scan_for_guidance(value, result: dict, path: Tuple[str, ...] = (), depth: int = 0) -> None:
    if depth > 8:
        return

    if isinstance(value, str):
        if not result.get("message") and value.strip():
            result["message"] = value.strip()[:500]
        for pattern in _TOOL_NAME_PATTERNS:
            match = pattern.search(value)
            if match:
                _record_candidate(result, match.group(1), path + ("<text>",))
                break

    elif isinstance(value, dict):
        for key in _TOOL_NAME_FIELDS:
            v = value.get(key)
            if isinstance(v, str):
                _record_candidate(result, v, path + (key,))
                break

        if not result.get("next_args"):
            for key in _ARGS_FIELDS:
                v = value.get(key)
                if isinstance(v, dict):
                    result["next_args"] = v
                    break

        if not result.get("message"):
            for key in ("message", "detail", "reason", "hint", "suggestion", "why"):
                v = value.get(key)
                if isinstance(v, str) and v.strip():
                    result["message"] = v.strip()[:500]
                    break

        for k, v in value.items():
            _scan_for_guidance(v, result, path + (str(k),), depth + 1)

    elif isinstance(value, list):
        for i, item in enumerate(value):
            _scan_for_guidance(item, result, path + (f"[{i}]",), depth + 1)


def _adjust_guidance_by_keywords(
    next_tool: str, message: str, base_type: str, source: str,
) -> str:
    next_tool_lower = (next_tool or "").lower()
    message_lower = (message or "").lower()

    if source == "error_next":
        return "mandatory"

    if any(kw in next_tool_lower for kw in _AUTH_FIX_TOOL_KEYWORDS):
        return "mandatory"

    if any(kw in message_lower for kw in _AUTH_ERROR_KEYWORDS):
        return "optional"

    if any(kw in message_lower for kw in _PARAM_ERROR_KEYWORDS):
        if any(kw in next_tool_lower for kw in _PARAM_FIX_TOOL_KEYWORDS):
            return "mandatory"
        return "optional"

    return base_type


def _pick_best_candidate(result: dict) -> Optional[dict]:
    candidates = result.get("_candidates") or []
    if not candidates:
        return None
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda x: (-x[1]["priority"], x[0]))
    return indexed[0][1]


def extract_guidance(payload: Any) -> Dict[str, Any]:
    """从任意 MCP 返回里提取引导信息。"""
    result: Dict[str, Any] = {
        "is_error": False,
        "next_tool": None,
        "next_args": {},
        "message": None,
        "guest_tools": [],
        "requires_auth": False,
        "guidance_type": "none",
        "guidance_path": (),
        "guidance_source": "none",
        "_candidates": [],
    }

    if not isinstance(payload, dict):
        result.pop("_candidates", None)
        return result

    result["is_error"] = bool(payload.get("isError"))

    structured = payload.get("structuredContent")
    content_blocks = payload.get("content") or []

    if isinstance(structured, dict):
        for key in _GUIDANCE_KEYS:
            if key in structured:
                _scan_for_guidance(structured[key], result, path=(key,))
        for key, value in structured.items():
            if key not in _GUIDANCE_KEYS:
                _scan_for_guidance(value, result, path=(key,))

        guest = structured.get("guest_tools")
        if isinstance(guest, dict):
            tools = guest.get("tools")
            if isinstance(tools, list):
                result["guest_tools"] = [t for t in tools if isinstance(t, str)]

    content_text = _first_text_block(content_blocks)
    if content_text:
        stripped = content_text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    _scan_for_guidance(parsed, result, path=("<content>",))
                    if not result["guest_tools"]:
                        inner_guest = parsed.get("guest_tools")
                        if isinstance(inner_guest, dict):
                            inner_tools = inner_guest.get("tools")
                            if isinstance(inner_tools, list):
                                result["guest_tools"] = [
                                    t for t in inner_tools if isinstance(t, str)
                                ]
            except json.JSONDecodeError:
                pass

        for pattern in _TOOL_NAME_PATTERNS:
            match = pattern.search(content_text)
            if match:
                _record_candidate(result, match.group(1), ("<content>", "<text>"))
                break

        if not result["message"]:
            result["message"] = content_text[:500]

    best = _pick_best_candidate(result)
    if best is not None:
        result["next_tool"] = best["tool"]
        result["guidance_path"] = best["path"]
        result["guidance_source"] = best["source"]

    result.pop("_candidates", None)

    msg_lower = (result["message"] or "").lower()
    result["requires_auth"] = any(kw in msg_lower for kw in _AUTH_KEYWORDS)

    if not result["next_tool"] and result["requires_auth"] and result["guest_tools"]:
        result["next_tool"] = result["guest_tools"][0]
        result["guidance_path"] = ("<guest_tools_fallback>",)
        result["guidance_source"] = "guest_tools_fallback"

    if result["next_tool"]:
        base_type = best["type"] if best else "optional"
        result["guidance_type"] = _adjust_guidance_by_keywords(
            next_tool=result["next_tool"],
            message=result["message"] or "",
            base_type=base_type,
            source=result["guidance_source"],
        )
    else:
        result["guidance_type"] = "none"

    return result