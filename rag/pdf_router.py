"""PDF 解析路由器：根据文本层决定走快路径还是 Docling。

设计：
    - 有文本层：Docling 版面分析（关闭 OCR）+ 保留 heading 层级
    - 无文本层：Docling + OCR
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


def has_text_layer(pdf_path: str, min_chars_per_page: int = 50) -> bool:
    """判断 PDF 是否有文本层。"""
    import fitz

    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        if total_pages == 0:
            return False
        total_chars = sum(len(page.get_text().strip()) for page in doc)
        avg = total_chars / total_pages
        logger.info(
            "[dsh][pdf] 文本层检查: %d 页, 平均 %.0f 字符/页 → %s",
            total_pages, avg, "有文本层" if avg >= min_chars_per_page else "扫描件",
        )
        return avg >= min_chars_per_page
    finally:
        doc.close()


def parse_pdf(pdf_path: str):
    """统一入口：解析 PDF，返回 chunk 列表。

    - 有文本层 → Docling 关闭 OCR（保留版面分析 + heading）
    - 无文本层 → Docling + OCR
    """
    use_text_layer = has_text_layer(pdf_path)

    from rag.docling_parser import _get_converter, _get_chunker, DoclingParser

    parser = DoclingParser()

    if use_text_layer:
        logger.info("[dsh][pdf] 走 Docling 版面分析（关闭 OCR）")
        chunks = _parse_with_text_layer(pdf_path, parser)
    else:
        logger.info("[dsh][pdf] 走 Docling + OCR")
        chunks = parser.parse_to_chunks(pdf_path)

    return chunks


def _parse_with_text_layer(pdf_path: str, parser):
    """有文本层路线：Docling 关闭 OCR，只用版面分析。

    速度：比全量 OCR 快 10-20 倍，比 PyMuPDF 快路径慢（保留版面分析）。
    """
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

    # ── 关键：do_ocr=False，直接用嵌入的文本层 ──
    pipeline_options = ThreadedPdfPipelineOptions(
        do_ocr=False,                  # 关闭 OCR
        ocr_batch_size=64,
        layout_batch_size=64,
        table_batch_size=4,
        generate_parsed_pages=True,    # 保留字体样式信号
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

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    import time
    t0 = time.perf_counter()
    result = converter.convert(pdf_path)
    elapsed = time.perf_counter() - t0
    logger.info("[dsh][pdf] Docling 版面分析完成，耗时 %.1fs", elapsed)

    doc = result.document

    # ── 用 HybridChunker 切块 ──
    chunker = _get_chunker()
    chunks = list(chunker.chunk(dl_doc=doc))

    logger.info("[dsh][pdf] 切块完成: %d 个 chunk", len(chunks))
    return chunks