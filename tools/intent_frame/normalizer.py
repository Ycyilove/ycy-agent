"""IntentNormalizer：把原始 goal 压成 IntentFrame。

规则优先，LLM 兜底（仅当规则置信度低时）。
"""
from __future__ import annotations

import logging
import re
from typing import List

from .frame import IntentFrame

logger = logging.getLogger(__name__)


_DOMAIN_KEYWORDS = {
    "news": ("新闻", "资讯", "报道", "实时", "最新", "动态", "热搜",
             "news", "latest", "realtime"),
    "sports": ("体育", "足球", "篮球", "球员", "比赛", "赛事", "运动员",
               "sports", "football", "basketball", "player", "match"),
    "knowledge": ("知识库", "文档", "资料", "概要", "摘要",
                  "knowledge", "document", "summary"),
    "file": ("文件", "目录", "路径", "csv", "pdf", "docx",
             "file", "directory", "path"),
    "code": ("代码", "运行", "执行", "python",
             "code", "run", "execute"),
    "web": ("网页", "页面", "网站", "网址", "链接",
            "web", "page", "website", "url"),
}

_FRESHNESS_KEYWORDS = {
    "realtime": ("实时", "现在", "当前", "此刻", "realtime", "now", "current"),
    "recent": ("最新", "最近", "近期", "这几天", "latest", "recent"),
}

_ACTION_KEYWORDS = {
    "search": ("搜索", "查找", "查", "找", "search", "find", "query"),
    "read": ("读取", "查看", "打开", "read", "view", "open"),
    "write": ("创建", "新建", "写入", "保存", "create", "write", "save"),
    "compute": ("计算", "统计", "compute", "calculate", "count"),
    "browse": ("浏览", "抓取", "fetch", "crawl", "extract"),
}


def _match(table, text: str) -> str:
    for key, kws in table.items():
        if any(k in text for k in kws):
            return key
    return ""


class IntentNormalizer:
    """规则优先 + 可选 LLM 兜底。"""

    def __init__(self, llm_provider=None):
        self._llm_provider = llm_provider
        self._llm = None

    def normalize(self, goal: str) -> IntentFrame:
        if not goal or not goal.strip():
            return IntentFrame(raw=goal, confidence=0.0)

        lower = goal.lower()
        domain = _match(_DOMAIN_KEYWORDS, lower) or "unknown"
        action = _match(_ACTION_KEYWORDS, lower) or "search"
        freshness = _match(_FRESHNESS_KEYWORDS, lower) or "any"
        entities = self._extract_entities(goal)

        conf = 0.8 if domain != "unknown" else 0.4
        if conf < 0.5 and self._llm_provider is not None:
            return self._llm_fallback(goal, entities)

        return IntentFrame(
            raw=goal, domain=domain, action=action,
            entities=entities, freshness=freshness,
            confidence=conf, source="rule",
        )

    @staticmethod
    def _extract_entities(goal: str) -> List[str]:
        try:
            from rag.entity.extractor import EntityExtractor
            return [m.surface for m in EntityExtractor().extract(goal)]
        except Exception:
            return re.findall(r"[\u4e00-\u9fff]{2,4}", goal)[:5]

    def _llm_fallback(self, goal: str, entities: List[str]) -> IntentFrame:
        try:
            if self._llm is None:
                self._llm = self._llm_provider()
            prompt = (
                "把用户问题归纳成结构化意图。只输出 JSON：\n"
                '{"domain": "news|sports|knowledge|file|code|web|unknown", '
                '"action": "search|read|write|compute|browse", '
                '"freshness": "realtime|recent|any"}\n\n'
                f"用户问题：{goal}"
            )
            raw = self._llm.generate(
                [{"role": "user", "content": prompt}], mode="quick",
            )
            import json as _json
            data = _json.loads((raw or "{}").strip().strip("`").strip())
            return IntentFrame(
                raw=goal,
                domain=data.get("domain", "unknown"),
                action=data.get("action", "search"),
                entities=entities,
                freshness=data.get("freshness", "any"),
                confidence=0.7, source="llm",
            )
        except Exception:
            logger.exception("[intent] LLM fallback failed")
            return IntentFrame(
                raw=goal, entities=entities, confidence=0.3, source="rule",
            )