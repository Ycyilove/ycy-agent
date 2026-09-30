"""LLM 驱动的查询改写模块。

把自然语言问题改写成适合向量检索的关键词组合，提升召回率。

设计：
    - 调用 ModelScopeLLM 的 quick 模式，低延迟
    - 失败时回退到原 query，不阻断主流程
    - 可选开关，由 RAG_QUERY_REWRITE_ENABLED 控制
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


_REWRITE_PROMPT = """你是一个搜索查询改写助手。把用户问题改写成适合向量检索的关键词组合。

规则：
1. 提取核心实体（人名、地名、赛事名、产品名等）
2. 补充可能的同义词/别名
3. 加入可能的答案形式（如"获奖名单"、"一等奖"、"名单"）
4. 只输出关键词，空格分隔，不要疑问句
5. 只输出改写结果，不要任何解释

示例：
- 输入："张三获奖了吗" → 输出："张三 获奖 奖项 一等奖 名单"
- 输入："第十届天梯赛讲了什么" → 输出："第十届 天梯赛 竞赛 内容 介绍"
- 输入："怎么创建产品" → 输出："创建产品 步骤 流程 教程"

用户问题：{query}
改写："""


class QueryRewriter:
    """把自然语言问题改写成检索关键词。"""

    def __init__(self, llm_provider=None):
        """
        Args:
            llm_provider: 返回 ModelScopeLLM 实例的 callable。
                          None 时惰性 import backend.services.llm。
        """
        self._llm_provider = llm_provider
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if self._llm_provider is not None:
            self._llm = self._llm_provider()
        else:
            from backend.services.llm import ModelScopeLLM
            self._llm = ModelScopeLLM()
        return self._llm

    def rewrite(self, query: str, timeout_hint: float = 10.0) -> str:
        """改写 query。失败时返回原 query。

        Args:
            query: 原始用户问题
            timeout_hint: 提示调用方预期超时（仅用于日志，不做强制超时）

        Returns:
            改写后的关键词字符串；失败时返回原 query
        """
        if not query or not query.strip():
            return query

        try:
            llm = self._get_llm()
            prompt = _REWRITE_PROMPT.format(query=query.strip())
            result = llm.generate(
                [{"role": "user", "content": prompt}],
                mode="quick",
            )
            rewritten = (result or "").strip()
            # 基本清洗：去掉引号、换行、markdown 包裹
            rewritten = rewritten.strip('"\'` \n')
            if "\n" in rewritten:
                rewritten = rewritten.split("\n")[0].strip()

            if not rewritten:
                return query

            logger.info(
                "[dsh][rewrite] %r → %r", query[:60], rewritten[:120]
            )
            return rewritten
        except Exception:
            logger.exception("[dsh][rewrite] 改写失败，用原 query")
            return query


# 全局单例
_default_rewriter: Optional[QueryRewriter] = None


def get_query_rewriter() -> QueryRewriter:
    global _default_rewriter
    if _default_rewriter is None:
        _default_rewriter = QueryRewriter()
    return _default_rewriter