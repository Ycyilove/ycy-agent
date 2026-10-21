"""快速 PDF 解析器：用 PyMuPDF 直接提取文本层，按字号推断层级。

适用：有文本层的 PDF（born-digital）。
速度：100 页文档几秒完成。
不适用：纯扫描件（无文本层），需要走 Docling + OCR。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional
import re

logger = logging.getLogger(__name__)


@dataclass
class FastChunk:
    """与 Docling chunk 结构对齐的最小字段集。"""
    text: str
    headings: List[str] = field(default_factory=list)
    heading_levels: List[int] = field(default_factory=list)
    page_no: Optional[int] = None
    font_size: float = 0.0
    is_bold: bool = False


# ── 标题层级推断：编号优先，字号兜底 ──

_NUM_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")

# 页眉/页脚/水印噪声：arXiv 编号、孤立页码、"Page N"、"第N页"
_NOISE_PATTERNS = (
    re.compile(r"^arxiv:\s*\d{4}\.\d{4,5}", re.I),
    re.compile(r"^page\s+\d+(\s*of\s*\d+)?$", re.I),
    re.compile(r"^第?\s*\d+\s*页$"),
    re.compile(r"^\d{1,4}$"),
)


def _is_noise_line(text: str) -> bool:
    t = text.strip()
    if len(t) <= 2:
        return True
    return any(p.match(t) for p in _NOISE_PATTERNS)


_LETTER_HEADING = re.compile(r"^[A-Z]\s+\S")

_TOP_SECTIONS = {
    "abstract", "摘要", "introduction", "引言",
    "references", "参考文献", "acknowledgments", "致谢",
    "appendix", "appendices", "附录",
}

def _infer_heading_level(text: str, ratio: float) -> int:
    t = text.strip()
    if t.lower() in _TOP_SECTIONS:
        return 1
    m = _NUM_HEADING.match(t)
    if m:
        return m.group(1).count(".") + 1
    if _LETTER_HEADING.match(t):
        return 1
    if ratio >= 1.5:
        return 1
    if ratio >= 1.3:
        return 2
    return 3


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

import re

# 页眉/页脚/水印噪声：arXiv 编号、孤立页码、"Page N"、"第N页"、"K-12"
_NOISE_PATTERNS = (
    re.compile(r"^arxiv:\s*\d{4}\.\d{4,5}", re.I),
    re.compile(r"^page\s+\d+(\s*of\s*\d+)?$", re.I),
    re.compile(r"^第?\s*\d+\s*页$"),
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^K-\d{1,4}$", re.I),
)

# 标题层级推断：编号优先
_NUM_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
_LETTER_HEADING = re.compile(r"^[A-Z]\s+\S")

# 无编号常见章节名 → 强制 level 1
_TOP_SECTIONS = {
    "abstract", "摘要", "introduction", "引言",
    "references", "参考文献", "acknowledgments", "致谢",
    "appendix", "appendices", "附录",
}


def _is_noise_line(text: str) -> bool:
    t = text.strip()
    if len(t) <= 2:
        return True
    return any(p.match(t) for p in _NOISE_PATTERNS)


def _infer_heading_level(text: str, ratio: float) -> int:
    """编号优先：'2'→1, '2.1'→2；无编号按字号比；常见章节名强制 1。"""
    t = text.strip()
    if t.lower() in _TOP_SECTIONS:
        return 1
    m = _NUM_HEADING.match(t)
    if m:
        return m.group(1).count(".") + 1
    if _LETTER_HEADING.match(t):
        return 1
    if ratio >= 1.5:
        return 1
    if ratio >= 1.3:
        return 2
    return 3

def _is_bold(flags: int) -> bool:
    """PyMuPDF span flags 的 bit 4 表示粗体。"""
    return bool(flags & (1 << 4))

def is_standalone_line(line_text: str, prev_line: str, next_line: str) -> bool:
    """判断一行是否单独成段。

    只依赖排版结构：
        - 前后是空行，或
        - 前后行与它字号/缩进不同（这里退化为：前后行为空）
    """
    return (not prev_line.strip()) or (not next_line.strip())

def _looks_like_heading(
    text: str,
    font_size: float,
    body_font_size: float,
    is_bold: bool,
    size_threshold: float = 1.15,
    standalone: bool = False,
) -> bool:
    """只依据排版信号判断标题。"""
    text = text.strip()
    if not text or _is_noise_line(text):
        return False

    # 主信号：字号显著更大，但不能是长句
    if font_size >= body_font_size * size_threshold:
        if text.endswith((".", "。")) and len(text) > 40:
            return False
        return True

    # 辅助信号：加粗 + 短 + 单独成段，且不能是句子
    if (
        is_bold
        and font_size >= body_font_size
        and len(text) <= 30
        and standalone
    ):
        if text.endswith((".", "。", "?", "？", "!", "！", ",", "，")):
            return False
        if sum(c in ",，;；:：" for c in text) >= 2:
            return False
        return True

    # 全大写短行（年报小标题常见）
    if text.isupper() and 5 <= len(text) <= 80:
        return True

    return False

def extract_tables(pdf_path: str) -> List[dict]:
    """用 PyMuPDF 的 find_tables() 识别表格。

    返回 [{"grid": [[cell_text, ...], ...], "page_numbers": [n]}]。
    识别不到时返回空列表，不影响段落 chunk。
    """
    import pymupdf

    out: List[dict] = []
    doc = pymupdf.open(pdf_path)
    try:
        for page_no, page in enumerate(doc, start=1):
            try:
                tabs = page.find_tables()
            except Exception:
                logger.exception("[dsh][fastpdf] find_tables 失败，跳过第 %d 页", page_no)
                continue
            for tab in getattr(tabs, "tables", []) or []:
                try:
                    rows = tab.extract()
                except Exception:
                    continue
                grid = [
                    [str(c).strip() if c is not None else "" for c in row]
                    for row in rows
                ]
                if len(grid) < 2 or len(grid[0]) < 2:
                    continue
                out.append({"grid": grid, "page_numbers": [page_no]})
    finally:
        doc.close()
    return out
    
def parse_fast(
    pdf_path: str,
    max_chars_per_chunk: int = 800,
    detect_headings: bool = True,
) -> List[FastChunk]:
    """快速解析 PDF，返回带 headings 的 chunk 列表。

    detect_headings=False 时不做 heading 识别，只按 max_chars_per_chunk 切正文，
    用于版面复杂（年报/杂志）且 Docling 失败时的安全降级。
    """
    import fitz
    from collections import Counter

    doc = fitz.open(pdf_path)
    try:
        # ── 第一遍：统计字号（只统计正文长度的 span，排除表格/页码/脚注）──
        size_counter: Counter = Counter()
        for page in doc:
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if len(text) >= 20:
                            size_counter[round(span["size"], 1)] += len(text)

        if not size_counter:
            # 退化为全量统计
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
            # 去重相邻重复 heading，并截断过深栈（>10 层视为异常）
            raw_headings = [h for h, _ in heading_stack][:10]
            raw_levels = [lv for _, lv in heading_stack][:10]
            dedup_h, dedup_lv = [], []
            for h, lv in zip(raw_headings, raw_levels):
                if not dedup_h or dedup_h[-1] != h:
                    dedup_h.append(h)
                    dedup_lv.append(lv)
            chunks.append(FastChunk(
                text=text,
                headings=dedup_h,
                heading_levels=dedup_lv,
                page_no=current_page,
                font_size=page_font_size,
            ))
            buffer_lines.clear()
            buffer_chars = 0

        for page_no, page in enumerate(doc, start=1):
            current_page = page_no
            d = page.get_text("dict")

            # 先把这一页的所有行拍平，便于取 prev/next 行。
            # 只依赖排版结构，不依赖文本内容。
            lines: List[dict] = []
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
                    lines.append({
                        "text": line_text,
                        "size": line_size,
                        "bold": line_bold,
                    })

            # ── 合并“编号行 + 标题行”：PyMuPDF 常把 "2.1\nModels" 拆成两行 ──
            merged: List[dict] = []
            j = 0
            while j < len(lines):
                cur = lines[j]
                nxt = lines[j + 1] if j + 1 < len(lines) else None
                is_num_only = bool(re.fullmatch(r"\d+(?:\.\d+)*", cur["text"]))
                is_letter_only = bool(re.fullmatch(r"[A-Z]", cur["text"]))
                if (
                    nxt is not None
                    and (is_num_only or is_letter_only)
                    and not re.match(r"^\d+(?:\.\d+)*\s+\S", nxt["text"])
                    and len(nxt["text"]) <= 80
                    and (nxt["bold"] or nxt["size"] >= body_font_size * 1.05)
                ):
                    merged.append({
                        "text": f"{cur['text']} {nxt['text']}",
                        "size": max(cur["size"], nxt["size"]),
                        "bold": cur["bold"] or nxt["bold"],
                    })
                    j += 2
                else:
                    merged.append(cur)
                    j += 1
            lines = merged

            # ── 合并"编号行 + 标题行"：PyMuPDF 常把 "2.1\nModels" 拆成两行 ──
            merged: List[dict] = []
            j = 0
            while j < len(lines):
                cur = lines[j]
                nxt = lines[j + 1] if j + 1 < len(lines) else None
                is_num_only = bool(re.fullmatch(r"\d+(?:\.\d+)*", cur["text"]))
                is_letter_only = bool(re.fullmatch(r"[A-Z]", cur["text"]))
                if (
                    nxt is not None
                    and (is_num_only or is_letter_only)
                    and not re.match(r"^\d+(?:\.\d+)*\s+\S", nxt["text"])
                    and len(nxt["text"]) <= 80
                    and (nxt["bold"] or nxt["size"] >= body_font_size * 1.05)
                ):
                    merged.append({
                        "text": f"{cur['text']} {nxt['text']}",
                        "size": max(cur["size"], nxt["size"]),
                        "bold": cur["bold"] or nxt["bold"],
                    })
                    j += 2
                else:
                    merged.append(cur)
                    j += 1
            lines = merged

            for i, line_info in enumerate(lines):
                line_text = line_info["text"]
                line_size = line_info["size"]
                line_bold = line_info["bold"]

                prev_line = lines[i - 1]["text"] if i > 0 else ""
                next_line = lines[i + 1]["text"] if i + 1 < len(lines) else ""

                if detect_headings:
                    is_heading = _looks_like_heading(
                        line_text, line_size, body_font_size, line_bold,
                        standalone=is_standalone_line(line_text, prev_line, next_line),
                    )
                else:
                    is_heading = False

                if is_heading:
                    ratio = line_size / body_font_size if body_font_size else 1.0
                    level = _infer_heading_level(line_text, ratio)
                    if heading_stack and level > heading_stack[-1][1] + 1:
                        level = heading_stack[-1][1] + 1
                    while heading_stack and heading_stack[-1][1] >= level:
                        heading_stack.pop()
                    # 先 flush 上一段正文，确保 heading 不混进正文
                    flush_buffer()
                    heading_stack.append((line_text, level))
                    page_font_size = line_size
                    continue

                # 无论是否 heading，文本都进入 buffer
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