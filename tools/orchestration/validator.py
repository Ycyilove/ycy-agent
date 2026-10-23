"""策略 B：决策前置校验。

判断"这个请求是否必须调用工具"。用于拦截 LLM 的"假装完成"。
"""

from __future__ import annotations
from typing import Optional, List

import re

from tools.semantics.tags import get_intent_tags


# 副作用动词
_SIDE_EFFECT_VERBS = (
    "创建", "新建", "写入", "写一个", "写个", "编辑", "修改", "改",
    "修复", "修正", "改正", "更正", "补充", "补齐", "增加", "添加",
    "移动", "重命名", "删除", "移除", "复制", "拷贝",
    "create", "write", "edit", "modify", "move", "rename",
    "delete", "remove", "copy", "mkdir",
    "fix", "repair", "patch", "correct", "update",
)

# 读取动词
_READ_VERBS = (
    "读取", "查看", "打开", "读取内容", "看", "显示", "列出", "列一下",
    "搜索", "查找", "找一下", "统计", "计算",
    "read", "view", "open", "show", "list", "search", "find",
    "count", "calculate",
)

# "教程"类表述
_TUTORIAL_PHRASES = (
    "怎么做", "如何", "教程", "命令是", "怎么用", "教我",
    "how to", "tutorial", "command",
)

_PATH_PATTERN = re.compile(r"[A-Za-z]:[\\/]|\.\w{1,5}\b|/[A-Za-z]")


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

    - action:    有副作用（创建/修改/删除/移动）
    - query:     只读（列出/读取/搜索/统计）
    - tutorial:  只是询问方法
    - smalltalk: 闲聊
    """
    if not goal:
        return "smalltalk"

    goal_lower = goal.lower()

    if any(p in goal_lower for p in _TUTORIAL_PHRASES):
        return "tutorial"

    has_path = bool(_PATH_PATTERN.search(goal))
    has_side_effect = any(v in goal_lower for v in _SIDE_EFFECT_VERBS)
    has_read = any(v in goal_lower for v in _READ_VERBS)

    if has_side_effect and (has_path or len(goal) > 5):
        return "action"
    if has_read and (has_path or "文件" in goal or "目录" in goal):
        return "query"

    return "smalltalk"


def requires_tool_call(goal: str) -> bool:
    """判断请求是否必须调用工具。"""
    kind = classify_request(goal)
    if kind in ("action", "query"):
        return True
    # 网页检索也强制要求调用工具
    if is_web_query(goal):
        return True
    return False


# ─────────────────────────────────────────────────────────────
# 网页检索完成度检查
# ─────────────────────────────────────────────────────────────

# 网页检索信号词
_WEB_SIGNALS = (
    "网页", "页面", "网站", "网址", "链接", "榜单", "排行榜",
    "新闻", "最新", "实时", "动态", "热搜",
    "web", "page", "website", "url", "link", "news", "latest",
)


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