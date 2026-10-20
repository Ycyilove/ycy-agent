"""Docling 适配层：替换 pypdf，提供带层级结构的文档解析。

设计：
    - 用 Docling 的版面分析模型直接从 PDF 读取标题层级
    - 用 ThreadedPdfPipelineOptions 做 OCR/版面批处理加速
    - 用 HybridChunker 做 token 感知的语义切块
    - chunk.meta.headings 是解析器原生输出的祖先路径，不需要正则猜测

参考：
    - Docling: https://github.com/docling-project/docling
    - HybridChunker: https://docling-project.github.io/docling/concepts/chunking/
"""

from __future__ import annotations

import logging
import os

# ── 必须在 import docling 之前设缓存路径 ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, "models"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"])
os.environ["HF_HUB_DISABLE_XET"] = "1"

from typing import Dict, List, Optional

# ── 强制禁用 HuggingFace Xet 存储后端 ──
# Xet 后端（cas-server.xethub.hf.co）在 hf-mirror.com 镜像下会返回 401，
# 禁用后 huggingface_hub 走传统 HTTP 下载，兼容镜像站。
os.environ["HF_HUB_DISABLE_XET"] = "1"

logger = logging.getLogger(__name__)


# ── 延迟初始化 Docling（首次 import 会加载模型，约 1-2 秒）──

_converter = None
_chunker = None

# ── OCR 预检：根据文本层自动决定是否启用 OCR ──

# 每页文本层字符数低于此阈值，视为"扫描页"
_TEXT_LAYER_MIN_CHARS = 50
# 扫描页占比超过此比例，才启用 OCR
_SCANNED_PAGE_RATIO_THRESHOLD = 0.3


def _detect_need_ocr(file_path: str) -> bool:
    """用 PyMuPDF 快速预检：是否需要 OCR。

    返回 True 表示"需要 OCR"（存在较多扫描页），
    返回 False 表示"不需要 OCR"（文本层充足）。
    """
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.warning("[dsh][docling] OCR 预检打开失败，回退到启用 OCR: %s", e)
        return True

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        return True

    scanned_pages = 0
    for page in doc:
        text = page.get_text()
        if len(text.strip()) < _TEXT_LAYER_MIN_CHARS:
            scanned_pages += 1

    doc.close()

    ratio = scanned_pages / total_pages
    need_ocr = ratio > _SCANNED_PAGE_RATIO_THRESHOLD

    logger.info(
        "[dsh][docling] OCR 预检: %d/%d 页文本层不足 (%.1f%%) → %s",
        scanned_pages, total_pages, ratio * 100,
        "启用 OCR" if need_ocr else "跳过 OCR",
    )
    return need_ocr


def _get_converter(file_path: str | None = None):
    """惰性创建 Docling DocumentConverter。

    file_path 用于 OCR 预检：如果文本层充足，则跳过 OCR 加速。
    注意：converter 是全局单例，首次创建时决定 do_ocr；
    如果同一进程内处理混合类型的 PDF，需要按文件重建 converter。
    """
    global _converter, _converter_do_ocr
    if _converter is not None:
        # 如果已有 converter，检查 do_ocr 是否需要切换
        if file_path is not None:
            need_ocr = _detect_need_ocr(file_path)
            if need_ocr != _converter_do_ocr:
                logger.info(
                    "[dsh][docling] do_ocr 需要从 %s 切换到 %s，重建 converter",
                    _converter_do_ocr, need_ocr,
                )
                _converter = None  # 触发重建
            else:
                return _converter
        else:
            return _converter

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.pipeline_options import (
        HeadingHierarchyOptions,
        ThreadedPdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # ── 决定是否启用 OCR ──
    do_ocr = _detect_need_ocr(file_path) if file_path else True

    pipeline_options = ThreadedPdfPipelineOptions(
        do_ocr=do_ocr,
        ocr_batch_size=64,
        layout_batch_size=64,
        table_batch_size=4,
        generate_parsed_pages=True,
    )
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.AUTO,
    )
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(
        enabled=True,
        use_bookmarks=True,
        use_numbering=True,
        use_style=True,
        bookmark_match_threshold=0.8,
    )

    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )
    _converter_do_ocr = do_ocr
    logger.info(
        "[dsh][docling] DocumentConverter created "
        "(do_ocr=%s, ocr_batch_size=64, threads=8, heading hierarchy enabled)",
        do_ocr,
    )
    return _converter


def _get_chunker():
    """惰性创建 HybridChunker（token 感知切块）。"""
    global _chunker
    if _chunker is not None:
        return _chunker

    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )
    from transformers import AutoTokenizer

    # 用通用 tokenizer 做 token 计数；换 bge 时再改
    tokenizer_id = os.getenv(
        "RAG_CHUNKER_TOKENIZER",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    max_tokens = int(os.getenv("RAG_CHUNKER_MAX_TOKENS", "256"))

    hf_tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_id),
        max_tokens=max_tokens,
    )
    _chunker = HybridChunker(
        tokenizer=hf_tokenizer,
        merge_peers=True,   # 合并同层级相邻段落，减少碎片
    )
    logger.info(
        "[dsh][docling] HybridChunker created (tokenizer=%s, max_tokens=%d)",
        tokenizer_id, max_tokens,
    )
    return _chunker


class DoclingParser:
    """Docling 解析器封装。"""

    def __init__(self):
        # 最近一次解析的 Docling Document（含 tables/lists），
        # 供 service.add_file 提取结构化 Record 用；快路径下为 None。
        self.last_document = None

    def parse_to_chunks(self, file_path: str):
        from rag.pdf_router import has_text_layer

        use_text_layer = has_text_layer(file_path)

        if use_text_layer:
            try:
                return self._parse_with_text_layer(file_path)
            except Exception:
                logger.exception(
                    "[dsh][docling] 关闭 OCR 路线失败，降级到 PyMuPDF 快路径"
                )
                from rag.pdf_fast_parser import parse_fast, extract_tables

                # 用 PyMuPDF 补表格结构，避免 docling 缺失时表格全丢
                try:
                    tables = extract_tables(file_path)
                except Exception:
                    logger.exception("[dsh][docling] PyMuPDF 表格识别失败")
                    tables = []

                self.last_fast_tables = tables

                fast_chunks = parse_fast(file_path)
                if fast_chunks:
                    return self._wrap_fast_chunks(fast_chunks)
                return self._parse_with_ocr(file_path)
        else:
            return self._parse_with_ocr(file_path)

    def _parse_with_ocr(self, file_path: str):
        """原有逻辑：Docling + OCR。"""
        converter = _get_converter(file_path)
        chunker = _get_chunker()
        result = converter.convert(file_path)
        self.last_document = result.document
        # ... 原有 parse_to_chunks 的剩余逻辑

    def _parse_with_text_layer(self, file_path: str):
        """有文本层：Docling 关闭 OCR，保留版面分析。"""
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
        from docling.datamodel.pipeline_options import (
            HeadingHierarchyOptions,
            ThreadedPdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = ThreadedPdfPipelineOptions(
            do_ocr=False,
            ocr_batch_size=64,
            layout_batch_size=64,
            table_batch_size=4,
            generate_parsed_pages=True,
        )
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.AUTO,
        )
        pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(
            enabled=True,
            use_bookmarks=True,
            use_numbering=True,
            use_style=True,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        import time
        t0 = time.perf_counter()
        result = converter.convert(file_path)
        logger.info(
            "[dsh][docling] 版面分析完成，耗时 %.1fs", time.perf_counter() - t0
        )

        doc = result.document
        self.last_document = doc
        chunker = _get_chunker()
        chunks = list(chunker.chunk(dl_doc=doc))
        for _c in chunks:
            if not hasattr(_c.meta, "heading_levels"):
                _c.meta.heading_levels = []
        logger.info("[dsh][docling] 切块完成: %d 个 chunk", len(chunks))
        return chunks
        
    @staticmethod
    def _wrap_fast_chunks(fast_chunks):
        """把 FastChunk 包装成和 Docling chunk 兼容的对象。"""
        class _Meta:
            def __init__(self, headings, heading_levels, page_numbers):
                self.headings = headings
                self.heading_levels = heading_levels
                self.page_numbers = page_numbers
                self.doc_items = []

        class _Chunk:
            def __init__(self, text, headings, heading_levels, page_numbers):
                self.text = text
                self.meta = _Meta(headings, heading_levels, page_numbers)

        return [
            _Chunk(
                c.text,
                c.headings,
                getattr(c, "heading_levels", []) or [],
                [c.page_no] if c.page_no is not None else [],
            )
            for c in fast_chunks
        ]
    
    def get_enriched_text(self, chunk) -> str:
        # 快路径的 chunk 没有 Docling 的 meta.doc_items，直接拼 headings
        headings = list(getattr(chunk.meta, "headings", None) or [])
        if headings:
            prefix = " > ".join(headings) + "\n"
            return prefix + chunk.text
        return chunk.text


# ── 便捷函数 ──

def parse_file_to_chunks(file_path: str):
    """从文件路径解析为 Docling chunk 列表。"""
    return DoclingParser().parse_to_chunks(file_path)