"""PDF 解析路由器：根据文本层决定走快路径还是 Docling。"""

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


def parse_pdf(pdf_path: str) -> List:
    """统一入口：解析 PDF，返回 chunk 列表。
    统一委托给 DoclingParser 处理。
    """
    from rag.docling_parser import DoclingParser
    logger.info("[dsh][pdf] 委托 DoclingParser.parse_to_chunks: %s", pdf_path)
    return DoclingParser().parse_to_chunks(pdf_path)