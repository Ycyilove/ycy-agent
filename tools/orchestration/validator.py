"""策略 B：决策前置校验。

判断"这个请求是否必须调用工具"。用于拦截 LLM 的"假装完成"。
"""

from __future__ import annotations

import re


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
    return kind in ("action", "query")