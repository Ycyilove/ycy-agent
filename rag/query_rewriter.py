"""LLM 驱动的查询改写模块。

设计原则：
    - 实体保护：原 query 中的实体必须原样出现在改写结果里，否则回退
    - 类型判断：纯实体 query 不改写，直接返回
    - 去重限长：防止 LLM 输出重复循环
    - 失败降级：任何异常都回退原 query

依赖：
    jieba.posseg（jieba 已随 BM25 安装，无新增依赖）
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional
from rag.text_utils import clean_cn_spaces

logger = logging.getLogger(__name__)


_REWRITE_PROMPT = """你是一个搜索查询改写助手。把用户问题改写成适合向量检索的关键词组合。

规则：
1. 提取核心实体（人名、地名、赛事名、产品名、机构名等）
2. **实体本身必须原样保留，不要扩展、不要加同义词、不要猜测身份**
3. 只对非实体部分（意图、关系、动作）补充同义词
4. 不要编造用户没有提到的信息
5. 只输出关键词，空格分隔，不要疑问句
6. 只输出改写结果，不要任何解释

示例：
- 输入："张三获奖了吗" → 输出："张三 获奖 奖项 名单"
- 输入："怎么创建产品" → 输出："创建产品 步骤 流程 教程"
- 输入："北京大学有哪些女生获奖" → 输出："北京大学 女生 获奖 名单"

用户问题：{query}
改写："""


# ────────── 实体提取（基于 jieba 词性标注，零新增依赖）──────────

# jieba 词性标注里代表实体的标签
_ENTITY_POS_FLAGS = frozenset({
    'nr',   # 人名
    'ns',   # 地名
    'nt',   # 机构名
    'nz',   # 其他专名
    'nrt',  # 音译人名
    'nrfg', # 人名（古）
    'nrf',  # 人名（姓）
})


def extract_entities(text: str) -> List[str]:
    """用 jieba.posseg 提取实体词。

    失败时返回空列表，调用方需处理。
    """
    if not text or not text.strip():
        return []
    try:
        import jieba.posseg as pseg
        entities = []
        for word, flag in pseg.cut(text):
            if flag in _ENTITY_POS_FLAGS and len(word) >= 2:
                entities.append(word)
        return entities
    except Exception:
        logger.exception("[dsh][rewrite] 实体提取失败")
        return []


def _normalize_for_match(text: str) -> str:
    """归一化：去掉空格和标点，用于实体子串匹配。"""
    return re.sub(r'[\s，。！？、；：（）【】《》""\'`]', '', text)


def entities_preserved(original: str, rewritten: str) -> bool:
    """检查原 query 的实体是否都保留在改写结果里。

    用归一化后的子串匹配，容忍标点差异。
    """
    entities = extract_entities(original)
    if not entities:
        return True  # 没提取到实体，视为通过

    rewritten_norm = _normalize_for_match(rewritten)
    original_norm = _normalize_for_match(original)

    for entity in entities:
        entity_norm = _normalize_for_match(entity)
        if not entity_norm:
            continue
        # 实体在改写结果里，或实体在原 query 里本就存在且改写结果包含它
        if entity_norm in rewritten_norm:
            continue
        # 特殊：实体本身太短（如2字人名），可能在改写时被拆开
        # 检查实体的每个字符是否都在改写结果里（宽松匹配）
        if all(ch in rewritten_norm for ch in entity_norm):
            continue
        logger.warning(
            "[dsh][rewrite] 实体 %r 在改写结果中丢失: %r → %r",
            entity, original, rewritten,
        )
        return False

    return True


# ────────── 主类 ──────────

class QueryRewriter:
    """把自然语言问题改写成检索关键词。"""

    # 纯实体 query 的长度阈值（归一化后）
    PURE_ENTITY_MAX_LEN = 6

    # 改写结果最大词数
    MAX_TOKENS = 8

    # 改写结果最大长度倍数（相对原 query）
    MAX_LENGTH_RATIO = 5.0

    def __init__(self, llm_provider=None):
        self._llm_provider = llm_provider
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if self._llm_provider is not None:
            self._llm = self._llm_provider()
        else:
            from core.llm.provider import ModelScopeLLM
            self._llm = ModelScopeLLM()
        return self._llm

    # ────────── 类型判断 ──────────

    def is_pure_entity(self, query: str) -> bool:
        """判断 query 是否只由实体构成（无意图词）。

        规则：
            - 归一化后长度 <= PURE_ENTITY_MAX_LEN
            - 且能被 jieba 切成 1-2 个实体，无其他词
        """
        stripped = _normalize_for_match(query)
        if not stripped or len(stripped) > self.PURE_ENTITY_MAX_LEN:
            return False

        entities = extract_entities(query)
        if not entities:
            return False

        # 去掉实体后的剩余部分应为空或仅标点
        remainder = stripped
        for e in entities:
            remainder = remainder.replace(_normalize_for_match(e), '', 1)
        return len(remainder) == 0

    # ────────── 主流程 ──────────

    def rewrite(self, query: str, timeout_hint: float = 10.0) -> str:
        """改写 query。失败或实体丢失时返回原 query。"""
        if not query or not query.strip():
            return query

        # ── 清洗 query 里的中文间空格 ──
        query = clean_cn_spaces(query)

        query = query.strip()

        # ── 1. 纯实体 query：跳过改写 ──
        if self.is_pure_entity(query):
            logger.info("[dsh][rewrite] 纯实体 query，跳过改写: %r", query)
            return query

        # ── 2. 调用 LLM 改写 ──
        try:
            llm = self._get_llm()
            prompt = _REWRITE_PROMPT.format(query=query)
            result = llm.generate(
                [{"role": "user", "content": prompt}],
                mode="quick",
            )
            rewritten = (result or "").strip()
        except Exception:
            logger.exception("[dsh][rewrite] LLM 改写失败，用原 query")
            return query

        # ── 3. 基础清洗 ──
        rewritten = rewritten.strip('"\'` \n')
        if "\n" in rewritten:
            rewritten = rewritten.split("\n")[0].strip()
        if not rewritten:
            return query

        # ── 4. 去重 + 限长 ──
        tokens = rewritten.split()
        seen = set()
        unique_tokens = []
        for t in tokens:
            if t and t not in seen:
                seen.add(t)
                unique_tokens.append(t)
        unique_tokens = unique_tokens[: self.MAX_TOKENS]
        rewritten = ' '.join(unique_tokens)

        # ── 5. 长度合理性检查 ──
        if len(rewritten) > len(query) * self.MAX_LENGTH_RATIO:
            logger.warning(
                "[dsh][rewrite] 改写过长，回退: %r → %r",
                query, rewritten,
            )
            return query

        # ── 6. 实体一致性检查（核心）──
        if not entities_preserved(query, rewritten):
            logger.warning(
                "[dsh][rewrite] 实体丢失，回退原 query: %r → %r",
                query, rewritten,
            )
            return query

        logger.info(
            "[dsh][rewrite] %r → %r", query[:60], rewritten[:120],
        )
        return rewritten


# ────────── 全局单例 ──────────

_default_rewriter: Optional[QueryRewriter] = None


def get_query_rewriter() -> QueryRewriter:
    global _default_rewriter
    if _default_rewriter is None:
        _default_rewriter = QueryRewriter()
    return _default_rewriter