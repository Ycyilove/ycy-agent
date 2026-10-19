"""Contextual Retrieval：给每个 chunk 生成情境化摘要。

设计：
    - 用 LLM 为每个 chunk 生成 50-100 字的情境化摘要
    - 摘要拼到 chunk 原文前面，作为向量嵌入和 BM25 的输入
    - 并发调用 + 单 chunk 超时 + 失败降级为原文本
    - 短 chunk（本身就带完整上下文）跳过，节省 LLM 调用

参考：
    - Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval

依赖：
    - core.llm.provider.ModelScopeLLM（SiliconFlow Qwen3-8B）
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


_CONTEXT_PROMPT = """请为以下文档片段生成一段简短的情境化摘要，说明它在整篇文档中的位置和主题。

要求：
1. 用 50-100 字描述：这段内容属于什么章节、讲的是什么
2. 不要复述原文内容，只提供"背景"
3. 不要编造原文没有的信息
4. 只输出摘要文本，不要任何前缀或解释

文档章节结构：
{headings}

文档片段原文：
{chunk_text}

情境化摘要："""


class ChunkContextualizer:
    """为 chunk 生成情境化摘要。"""

    def __init__(
        self,
        llm_provider=None,
        max_workers: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Args:
            llm_provider: 返回 LLM 实例的 callable；默认用 ModelScopeLLM
            max_workers: 并发线程数
            min_chunk_chars: 短于此字符数的 chunk 跳过摘要（本身已带上下文）
            timeout: 单 chunk LLM 调用超时（秒）
        """
        if max_workers is None:
            max_workers = int(os.getenv("RAG_CONTEXTUAL_CONCURRENCY", "5"))
        if min_chunk_chars is None:
            min_chunk_chars = int(os.getenv("RAG_CONTEXTUAL_MIN_CHARS", "80"))
        if timeout is None:
            timeout = float(os.getenv("RAG_CONTEXTUAL_TIMEOUT", "30"))

        self.max_workers = max_workers
        self.min_chunk_chars = min_chunk_chars
        self.timeout = timeout
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

    @staticmethod
    def _build_headings_str(headings: List[str]) -> str:
        if not headings:
            return "（无明确章节）"
        return " > ".join(headings)

    def _contextualize_one(self, chunk_text: str, headings: List[str]) -> str:
        """为单个 chunk 生成摘要。失败返回空字符串（调用方降级）。"""
        prompt = _CONTEXT_PROMPT.format(
            headings=self._build_headings_str(headings),
            chunk_text=chunk_text[:2000],   # 截断，避免超长
        )
        try:
            result = self._get_llm().generate(
                [{"role": "user", "content": prompt}],
                mode="quick",
            )
            summary = (result or "").strip()
            # 去掉可能的引号、前缀
            summary = summary.strip('"\'` ')
            # 单行化
            if "\n" in summary:
                summary = summary.split("\n")[0].strip()
            # 长度合理性
            if len(summary) < 10 or len(summary) > 300:
                logger.debug(
                    "[dsh][ctx] 摘要长度不合理(%d)，丢弃", len(summary)
                )
                return ""
            return summary
        except Exception as e:
            logger.debug("[dsh][ctx] chunk 摘要生成失败: %s", e)
            return ""

    def contextualize_batch(
        self,
        chunks: List[Dict],
    ) -> List[Dict]:
        """批量生成情境化摘要。

        Args:
            chunks: 每个元素形如
                {"text": "原 chunk 文本", "headings": ["H1", "H2"]}

        Returns:
            与输入等长的列表，每个元素形如
                {"enriched_text": "摘要\\n\\n原文", "summary": "..."}
            失败或跳过的 chunk，enriched_text 等于原文本。
        """
        results: List[Optional[Dict]] = [None] * len(chunks)
        tasks = []   # (idx, text, headings)

        for i, c in enumerate(chunks):
            text = (c.get("text") or "").strip()
            headings = c.get("headings") or []
            if len(text) < self.min_chunk_chars:
                # 短 chunk 直接跳过摘要
                results[i] = {"enriched_text": text, "summary": ""}
                continue
            tasks.append((i, text, headings))

        if not tasks:
            logger.info("[dsh][ctx] 所有 chunk 均短于阈值，跳过摘要生成")
            return results   # type: ignore

        logger.info(
            "[dsh][ctx] 开始生成情境化摘要：%d 个 chunk（并发=%d, 超时=%.0fs）",
            len(tasks), self.max_workers, self.timeout,
        )

        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {
                ex.submit(self._contextualize_one, text, headings): (idx, text)
                for idx, text, headings in tasks
            }
            for future in as_completed(future_map):
                idx, original_text = future_map[future]
                try:
                    summary = future.result(timeout=self.timeout)
                except Exception as e:
                    logger.debug("[dsh][ctx] future 异常: %s", e)
                    summary = ""

                if summary:
                    enriched = f"{summary}\n\n{original_text}"
                else:
                    enriched = original_text

                results[idx] = {"enriched_text": enriched, "summary": summary}
                completed += 1
                if completed % 50 == 0 or completed == len(tasks):
                    logger.info(
                        "[dsh][ctx] 进度 %d/%d", completed, len(tasks)
                    )

        # 兜底：不可能为 None，但保险
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {
                    "enriched_text": chunks[i].get("text", ""),
                    "summary": "",
                }

        n_ok = sum(1 for r in results if r and r.get("summary"))
        logger.info(
            "[dsh][ctx] 完成：%d/%d 成功生成摘要",
            n_ok, len(results),
        )
        return results   # type: ignore