"""反向兜底搜索：模型承认信息缺失时，走 core.ports.SearchPort 补一次。

B3 起不再直接 import tools.mcp_discovery，改为走端口：
    - SearchPort 已注入 → 调 search_web
    - 未注入 → 返回 None，answer_stream 跳过兜底（不抛错）
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_MISSING_INFO_PHRASES = (
    "没有权限访问", "无法访问", "没有相关信息",
    "无法提供", "建议通过官方渠道", "建议您通过官方",
    "我无法", "我不知道", "没有足够的信息",
    "无法获取", "不在我的知识",
)

_SEARCH_EVIDENCE_HINTS = ("tavily", "fetch", "search", "knowledge")


def answer_admits_missing(answer_text: str) -> bool:
    lower = (answer_text or "").lower()
    return any(p in lower for p in _MISSING_INFO_PHRASES)


def has_search_evidence(tool_context: Optional[List[str]]) -> bool:
    for t in (tool_context or []):
        if "返回" in t and any(h in t.lower() for h in _SEARCH_EVIDENCE_HINTS):
            return True
    return False


def should_fallback(
    mode: str,
    answer_text: str,
    tool_context: Optional[List[str]],
) -> bool:
    """只有 ask 模式 + 承认缺信息 + 本 turn 没搜过 → 才兜底。"""
    from core.task.modes import MODE_ASK
    if mode != MODE_ASK:
        return False
    if not answer_admits_missing(answer_text):
        return False
    if has_search_evidence(tool_context):
        return False
    return True


async def run_fallback(
    message: str,
    agent_provider,
) -> Optional[List[str]]:
    """走 SearchPort 兜底搜索。未注入 / 失败返回 None。"""
    from core import ports

    port = ports.get_search_port()
    if port is None:
        logger.info(
            "[dsh][answer] fallback skipped: SearchPort 未注入"
        )
        return None

    try:
        text = await port.search_web(message, max_results=5)
    except Exception:
        logger.exception("[dsh][answer] SearchPort.search_web failed")
        return None

    if not text:
        return None
    return [f"【兜底搜索结果】\n{text}"]


__all__ = [
    "answer_admits_missing",
    "has_search_evidence",
    "should_fallback",
    "run_fallback",
]