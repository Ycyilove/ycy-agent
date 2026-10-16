"""快速 PDF 解析器：用 PyMuPDF 直接提取文本层，按字号推断层级。

适用：有文本层的 PDF（born-digital）。
速度：100 页文档几秒完成。
不适用：纯扫描件（无文本层），需要走 Docling + OCR。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FastChunk:
    """与 Docling chunk 结构对齐的最小字段集。"""
    text: str
    headings: List[str] = field(default_factory=list)
    page_no: Optional[int] = None
    font_size: float = 0.0
    is_bold: bool = False


def has_text_layer(pdf_path: str, min_chars_per_page: int = 50) -> bool:
    """判断 PDF 是否有文本层。

    逐页统计字符数，平均每页低于阈值视为扫描件。
    """
    import fitz

    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        if total_pages == 0:
            return False

        total_chars = 0
        for page in doc:
            total_chars += len(page.get_text().strip())

        avg = total_chars / total_pages
        logger.info(
            "[dsh][fastpdf] 文本层检查: %d 页, 平均 %.0f 字符/页 → %s",
            total_pages, avg, "有文本层" if avg >= min_chars_per_page else "扫描件",
        )
        return avg >= min_chars_per_page
    finally:
        doc.close()


def _is_bold(flags: int) -> bool:
    """PyMuPDF span flags 的 bit 4 表示粗体。"""
    return bool(flags & (1 << 4))


def _looks_like_heading(
    text: str,
    font_size: float,
    body_font_size: float,
    is_bold: bool,
    size_threshold: float = 1.15,
) -> bool:
    """判断某行是否是标题。

    规则（任一满足）：
        1. 字号 >= 正文字号 × size_threshold
        2. 加粗 + 字号 >= 正文字号
        3. 文本较短 + 加粗
    """
    text = text.strip()
    if not text:
        return False
    if len(text) > 60:      # 标题通常不会太长
        return False
    if font_size >= body_font_size * size_threshold:
        return True
    if is_bold and font_size >= body_font_size:
        return True
    if is_bold and len(text) <= 30:
        return True
    return False


def parse_fast(
    pdf_path: str,
    max_chars_per_chunk: int = 800,   # 新增参数
) -> List[FastChunk]:
    """快速解析 PDF，返回带 headings 的 chunk 列表。

    新增：正文累积到 max_chars_per_chunk 就主动 flush，
    避免整份文档被压成几个巨大 chunk。
    """
    import fitz
    from collections import Counter

    doc = fitz.open(pdf_path)
    try:
        # ── 第一遍：统计字号 ──
        size_counter: Counter = Counter()
        for page in doc:
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            size_counter[round(span["size"], 1)] += len(text)

        if not size_counter:
            logger.warning("[dsh][fastpdf] 未提取到任何文本")
            return []

        body_font_size = size_counter.most_common(1)[0][0]
        logger.info("[dsh][fastpdf] 正文字号: %.1f", body_font_size)

        # ── 第二遍：逐 span 遍历 ──
        chunks: List[FastChunk] = []
        heading_stack: List[tuple] = []
        buffer_lines: List[str] = []
        buffer_chars = 0                    # 新增：buffer 当前字符数
        current_page = None
        page_font_size = body_font_size

        def flush_buffer():
            nonlocal buffer_chars
            if not buffer_lines:
                return
            text = "\n".join(buffer_lines).strip()
            if not text:
                buffer_lines.clear()
                buffer_chars = 0
                return
            headings = [h for h, _ in heading_stack]
            chunks.append(FastChunk(
                text=text,
                headings=headings,
                page_no=current_page,
                font_size=page_font_size,
            ))
            buffer_lines.clear()
            buffer_chars = 0

        for page_no, page in enumerate(doc, start=1):
            current_page = page_no
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    line_text = ""
                    line_size = 0.0
                    line_bold = False
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        line_size = max(line_size, span.get("size", 0))
                        if _is_bold(span.get("flags", 0)):
                            line_bold = True
                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    if _looks_like_heading(
                        line_text, line_size, body_font_size, line_bold
                    ):
                        flush_buffer()
                        ratio = line_size / body_font_size
                        if ratio >= 1.5:
                            level = 1
                        elif ratio >= 1.3:
                            level = 2
                        else:
                            level = 3
                        while heading_stack and heading_stack[-1][1] >= level:
                            heading_stack.pop()
                        heading_stack.append((line_text, level))
                    else:
                        # ── 新增：超长主动切分 ──
                        if buffer_chars + len(line_text) > max_chars_per_chunk:
                            flush_buffer()
                        page_font_size = line_size
                        buffer_lines.append(line_text)
                        buffer_chars += len(line_text)

        flush_buffer()

        logger.info(
            "[dsh][fastpdf] 解析完成: %d 个 chunk, %d 个 heading 层级",
            len(chunks), len(set(h for _, h in heading_stack)),
        )
        return chunks
    finally:
        doc.close()