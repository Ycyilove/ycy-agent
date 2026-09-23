"""意图路由：从 goal 快速判定应该用哪个工具。

返回的 tool_name 必须存在于 tools 里。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"']*|/[^\s\"']+")


def _extract_path(goal: str) -> Optional[str]:
    m = _PATH_RE.search(goal)
    return m.group(0) if m else None


def _has_path(goal: str) -> bool:
    return _extract_path(goal) is not None


def _pick(tools: Dict[str, Any], *suffixes: str) -> Optional[str]:
    """按后缀优先级在 tools 里挑一个；找不到返回 None。"""
    for suffix in suffixes:
        for name in tools:
            if name.endswith(suffix):
                return name
    return None


def route_intent(goal: str, tools: Dict[str, Any]) -> Optional[str]:
    """根据 goal 意图选出工具名。找不到合适的返回 None。"""
    lower = goal.lower()
    has_path = _has_path(goal)

    # ── 创建/写入 ──
    if any(k in goal for k in ("创建", "新建", "写入", "写文件", "保存", "生成文件")) \
       or any(k in lower for k in ("create file", "write file", "save file")):
        if has_path:
            picked = _pick(tools, "write_file", "create_file", "write_text_file")
            if picked:
                logger.info("[dsh][intent] create-file → %s", picked)
                return picked

    # ── 读取 ──
    if any(k in goal for k in ("读取", "查看", "打开", "看看", "内容")) \
       or any(k in lower for k in ("read file", "open file", "cat ")):
        if has_path:
            picked = _pick(tools, "read_text_file", "read_file")
            if picked:
                logger.info("[dsh][intent] read-file → %s", picked)
                return picked

    # ── 列出目录 ──
    if any(k in goal for k in ("列出", "目录内容", "文件夹", "列出目录")) \
       or any(k in lower for k in ("list directory", "list dir", "ls ")):
        picked = _pick(tools, "list_directory", "list_files", "directory_tree")
        if picked:
            logger.info("[dsh][intent] list-dir → %s", picked)
            return picked

    # ── 搜索文件 ──
    if any(k in goal for k in ("搜索文件", "查找文件", "找文件")) \
       or any(k in lower for k in ("search file", "find file")):
        picked = _pick(tools, "search_files", "search_knowledge", "find_files")
        if picked:
            logger.info("[dsh][intent] search → %s", picked)
            return picked

    # ── 删除/移动 ──
    if any(k in goal for k in ("删除", "移除", "删掉")) \
       or any(k in lower for k in ("delete file", "remove file")):
        picked = _pick(tools, "delete_file", "remove_file", "move_file")
        if picked:
            logger.info("[dsh][intent] delete → %s", picked)
            return picked

    if any(k in goal for k in ("重命名", "改名", "移动")) \
       or any(k in lower for k in ("rename", "move file")):
        picked = _pick(tools, "rename_file", "move_file")
        if picked:
            logger.info("[dsh][intent] rename/move → %s", picked)
            return picked

    # ── 时间 ──
    if any(k in goal for k in ("现在几点", "当前时间", "今天几号", "时间戳")) \
       or any(k in lower for k in ("current time", "now is", "what time")):
        picked = _pick(tools, "get_current_time", "current_time")
        if picked:
            logger.info("[dsh][intent] time → %s", picked)
            return picked

    # ── 默认：无法判定 ──
    return None

# ── 兼容旧接口 ──

INTENT_TO_TAG: Dict[str, str] = {
    # 老代码可能用到，提供一个最小映射
    "read": "read",
    "write": "write",
    "list": "list",
    "search": "search",
    "delete": "delete",
    "move": "move",
    "time": "time",
}


def extract_intent(goal: str) -> Optional[str]:
    """根据 goal 提取粗粒度意图标签。

    老接口的兼容实现。返回 'read' / 'write' / 'list' / 'search' /
    'delete' / 'move' / 'time' / None。
    """
    lower = goal.lower()
    if any(k in goal for k in ("创建", "新建", "写入", "写文件", "保存", "生成文件")):
        return "write"
    if any(k in lower for k in ("create file", "write file", "save file")):
        return "write"

    if any(k in goal for k in ("读取", "查看", "打开", "看看", "内容")):
        return "read"
    if any(k in lower for k in ("read file", "open file", "cat ")):
        return "read"

    if any(k in goal for k in ("列出", "目录", "文件夹")):
        return "list"
    if any(k in lower for k in ("list directory", "list dir", "ls ")):
        return "list"

    if any(k in goal for k in ("搜索", "查找", "找文件")):
        return "search"
    if any(k in lower for k in ("search file", "find file")):
        return "search"

    if any(k in goal for k in ("删除", "移除", "删掉")):
        return "delete"
    if any(k in lower for k in ("delete file", "remove file")):
        return "delete"

    if any(k in goal for k in ("重命名", "改名", "移动")):
        return "move"
    if any(k in lower for k in ("rename", "move file")):
        return "move"

    if any(k in goal for k in ("现在几点", "当前时间", "今天几号", "时间戳")):
        return "time"
    if any(k in lower for k in ("current time", "what time")):
        return "time"

    return None