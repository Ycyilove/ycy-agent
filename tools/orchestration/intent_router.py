"""策略 D：意图-工具直连。

在 LLM 决策之前，先尝试用规则匹配意图，直接选工具。
命中则跳过 LLM 决策（提速 + 提准）。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# 意图 → 工具用途标签
INTENT_TO_TAG: Dict[str, str] = {
    "read_dir": "read_dir",
    "read_text": "read_text",
    "read_multiple": "read_text",
    "traverse": "traverse_dir",
    "search_name": "search_name",
    "search_content": "search_content",
    "stat": "stat",
    "mkdir": "mkdir",
    "write_new": "write_new",
    "write_modify": "write_modify",
    "move": "move",
}


_PATH_PATTERN = re.compile(r"[A-Za-z]:[\\/][^\s，。；,;]+")


def _has_path(goal: str) -> bool:
    return bool(_PATH_PATTERN.search(goal))


def extract_intent(goal: str) -> Optional[str]:
    """从用户目标里提取意图类别。

    只在简单、明确的场景下返回；不确定时返回 None，让 LLM 决策。
    """
    if not goal:
        return None

    g = goal.lower()

    # 目录类
    if any(kw in goal for kw in ("目录树", "完整目录", "目录结构", "traverse")):
        return "traverse"
    if any(kw in goal for kw in ("列出", "列一下", "查看目录", "list dir")):
        if _has_path(goal) or "目录" in goal:
            return "read_dir"

    # 读文件
    if any(kw in goal for kw in ("同时读取", "多个文件", "读取多个")):
        return "read_multiple"
    if any(kw in goal for kw in ("读取", "打开", "内容", "看看")):
        if _has_path(goal) or re.search(r"\.\w{1,5}\b", goal):
            return "read_text"

    # 创建目录
    if any(kw in goal for kw in ("创建目录", "新建目录", "建一个目录", "mkdir")):
        return "mkdir"

    # 创建文件
    if any(kw in goal for kw in ("创建文件", "新建文件", "写一个文件", "写个文件")):
        return "write_new"

    # 修改文件
    if any(kw in goal for kw in ("修改", "编辑", "替换", "改成", "改为")):
        if _has_path(goal) or re.search(r"\.\w{1,5}\b", goal):
            return "write_modify"

    # 移动
    if any(kw in goal for kw in ("移动到", "移到", "重命名", "改名", "rename")):
        return "move"

    # 搜索
    if any(kw in goal for kw in ("搜索内容", "内容包含", "包含")):
        return "search_content"
    if any(kw in goal for kw in ("搜索", "查找", "找一下")):
        return "search_name"

    # 文件信息
    if any(kw in goal for kw in ("文件信息", "文件大小", "修改时间", "file info")):
        return "stat"

    return None


# 意图 → 优先使用的工具基名（用于从候选里筛）
_INTENT_PREFERRED_TOOLS: Dict[str, List[str]] = {
    "read_dir": ["list_directory"],
    "read_text": ["read_text_file", "read_file"],
    "read_multiple": ["read_multiple_files"],
    "traverse": ["directory_tree"],
    "search_name": ["search_files"],
    "search_content": ["search_files"],
    "stat": ["get_file_info"],
    "mkdir": ["create_directory"],
    "write_new": ["write_file"],
    "write_modify": ["edit_file"],
    "move": ["move_file"],
}


def route_intent(
    goal: str,
    available_tools: Dict[str, Any],
) -> Optional[str]:
    """尝试用规则匹配到工具名。返回 None 表示不命中，落回 LLM 决策。"""
    intent = extract_intent(goal)
    if not intent:
        return None

    preferred = _INTENT_PREFERRED_TOOLS.get(intent, [])
    if not preferred:
        return None

    # 从可用工具里找带对应 basename 的工具
    for tool_name in available_tools.keys():
        basename = tool_name.rsplit("__", 1)[-1]
        if basename in preferred:
            return tool_name

    return None