"""策略 B：决策前置校验。

判断"这个请求是否必须调用工具"。用于拦截 LLM 的"假装完成"。
"""

from __future__ import annotations
from typing import Optional, List

import re

from tools.semantics.tags import get_intent_tags


# ── MCP 工具名兜底 tag 推断 ──
# registry 里 MCP 工具可能没有 tags（tags.py 只覆盖内置工具基名）。
# 这里按 remote_name 的后缀/子串推断一个粗 tag，供完成度校验用。
_MCP_NAME_TAG_RULES = (
    (("search", "query", "find"), "search_name"),
    (("news", "feed", "latest", "realtime"), "network_read"),
    (("extract", "fetch", "crawl", "read", "get"), "network_read"),
    (("research", "analyze"), "network_read"),
    (("write", "create", "save", "edit", "update"), "write_new"),
    (("delete", "remove"), "write_new"),
    (("list", "dir", "tree"), "read_dir"),
)


def _mcp_tool_fallback_tags(tool_name: str) -> List[str]:
    """给 mcp__<server>__<remote> 工具推断粗 tag。"""
    if not tool_name or not tool_name.startswith("mcp__"):
        return []
    parts = tool_name.split("__", 2)
    if len(parts) != 3:
        return []
    remote = parts[2].lower()
    tags: List[str] = []
    for keywords, tag in _MCP_NAME_TAG_RULES:
        if any(k in remote for k in keywords):
            tags.append(tag)
    return tags


def _tool_tags(tool_name: str) -> List[str]:
    """先查 tags.py 的显式映射，空则用 MCP 兜底推断。"""
    tags = get_intent_tags(tool_name)
    if tags:
        return tags
    return _mcp_tool_fallback_tags(tool_name)


def classify_request(goal: str) -> str:
    """把请求分类成 action / query / tutorial / smalltalk。

    改为调用 tools.semantics.intent.classify（单一真源）。
    保留原返回字符串，兼容下游调用方。
    """
    from tools.semantics import (
        classify_intent,
        INTENT_GENERATE, INTENT_MODIFY, INTENT_READ, INTENT_QUERY,
        INTENT_DB, INTENT_CODE, INTENT_TUTORIAL,
    )

    intent = classify_intent(goal or "")
    primary = intent.primary

    if primary in (INTENT_GENERATE, INTENT_MODIFY, INTENT_CODE):
        return "action"
    if primary in (INTENT_READ, INTENT_QUERY, INTENT_DB):
        return "query"
    if primary == INTENT_TUTORIAL:
        return "tutorial"
    return "smalltalk"


def requires_tool_call(goal: str) -> bool:
    """判断请求是否必须调用工具。

    改为调用 tools.semantics.intent.classify（单一真源）。
    只要 primary 是任何非 tutorial/chitchat 的意图，就要求工具调用。
    """
    from tools.semantics import (
        classify_intent,
        INTENT_GENERATE, INTENT_MODIFY, INTENT_READ, INTENT_QUERY,
        INTENT_DB, INTENT_CODE,
    )

    intent = classify_intent(goal or "")
    if intent.primary in {
        INTENT_GENERATE, INTENT_MODIFY, INTENT_READ,
        INTENT_QUERY, INTENT_DB, INTENT_CODE,
    }:
        return True

    # 外部事实查询（保留原结构化信号，不走 intent.py）
    if _looks_like_external_fact_query(goal):
        return True

    return False

# ── 结构化信号：外部事实查询 ──
# 不用“宕机/故障/事件”这类会不断膨胀的词表，
# 而用「外部实体 + 时间/历史词 + 疑问句式」的组合信号。
_EXTERNAL_ENTITY_HINTS = (
    "deepseek", "openai", "claude", "anthropic", "google", "microsoft",
    "meta", "xai", "grok", "qwen", "通义", "文心", "豆包", "kimi",
    "公司", "产品", "平台", "服务", "模型", "官网",
)
_HISTORY_TIME_HINTS = (
    "曾经", "历史", "所有", "发生过", "出现过", "哪些", "几次",
    "过去", "之前", "以前", "至今", "以来",
)
_FACT_QUESTION_HINTS = (
    "吗", "？", "?", "什么", "哪些", "多少", "是否", "有没有",
)


def _looks_like_external_fact_query(goal: str) -> bool:
    if not goal:
        return False
    g = goal.lower()
    has_entity = any(k in g for k in _EXTERNAL_ENTITY_HINTS)
    has_time = any(k in g for k in _HISTORY_TIME_HINTS)
    has_question = any(k in g for k in _FACT_QUESTION_HINTS)
    # 三选二即可：外部实体 + 时间词，或外部实体 + 疑问句
    return has_entity and (has_time or has_question)


# ─────────────────────────────────────────────────────────────
# 网页检索完成度检查
# ─────────────────────────────────────────────────────────────


def is_web_query(goal: str) -> bool:
    """判断请求是否属于网页检索类。

    与 classify_request 正交：一个请求可能同时是 query 和 web_query。
    """
    if not goal:
        return False
    goal_lower = goal.lower()
    return any(s in goal_lower for s in _WEB_SIGNALS)


def check_web_query_completion(state) -> Optional[str]:
    """网页检索类请求的完成度检查。

    返回拒绝原因（字符串）；通过则返回 None。

    完成条件（用 intent tags 判断，不依赖工具名字符串）：
        1. 至少调用过一次「搜索类」工具（tags 含 search_name）
        2. 至少调用过一次「抓取/读取类」工具（tags 含 network_read）
    """
    seq = getattr(state, "successful_tool_sequence", []) or []

    if not seq:
        return (
            "⛔ 校验失败 —— 这是网页检索请求，但还没有成功调用任何工具。\n"
            "**下一步：先用搜索类工具获取候选 URL。**"
        )

    has_search = False
    has_extract = False
    for name in seq:
        tags = _tool_tags(name)
        if "search_name" in tags or "search_content" in tags:
            has_search = True
        if "network_read" in tags:
            has_extract = True
        if has_search and has_extract:
            break

    if not has_search:
        return (
            "⛔ 校验失败 —— 网页检索必须先调用搜索类工具获取候选 URL。\n"
            "**下一步：调用搜索类工具（如 search / query 类 MCP 工具）。**"
        )

    if not has_extract:
        return (
            "⛔ 校验失败 —— 你只做了搜索，没有读取搜索结果里的 URL。\n"
            "用户问的是页面内容，不是搜索摘要。\n"
            "**下一步：从搜索结果里选一个最相关的 URL，"
            "用 extract / fetch / crawl 类工具读取页面内容。**"
        )

    return None