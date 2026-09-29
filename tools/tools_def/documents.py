"""二进制文档处理工具（PDF / XLSX / DOCX）。

设计：
    - 读类工具 danger_level=safe，写类=medium
    - 复用 file.py 的 is_safe_path 路径保护
    - 不依赖 pandas：Excel 用 openpyxl，PDF 用 pypdf，DOCX 用 python-docx
    - 所有写操作返回 new_path，方便 orchestrator 记录 file_log
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import tool


logger = logging.getLogger(__name__)


# ── 复用 file.py 的路径保护 ──
try:
    from .file import is_safe_path
except ImportError:
    def is_safe_path(path: str, base_dir: str = None) -> bool:  # type: ignore
        return True


def _check_path(path: str) -> Optional[Dict[str, Any]]:
    """路径安全检查，返回 None 表示通过。"""
    if not os.path.exists(path):
        return {"success": False, "error": f"文件不存在: {path}"}
    if not is_safe_path(os.path.abspath(path)):
        return {"success": False, "error": f"操作被拒绝：路径受保护: {path}"}
    return None


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


# ═════════════════════════════════════════════════════════════
# PDF
# ═════════════════════════════════════════════════════════════

@tool(
    name="read_pdf_text",
    description=(
        "读取 PDF 文件的文本内容，支持指定页码范围。"
        "用于「这个 PDF 讲了什么」「提取第 3-5 页内容」这类问题。"
    ),
    parameters={
        "file_path": {"type": "str", "description": "PDF 文件路径（必填）"},
        "start_page": {"type": "int", "description": "起始页码（从 1 开始），默认 1"},
        "end_page": {"type": "int", "description": "结束页码（含），默认读全部"},
        "max_chars": {"type": "int", "description": "单次返回最大字符数，默认 50000"},
    },
    examples=[
        "read_pdf_text(file_path='D:/a/paper.pdf')",
        "read_pdf_text(file_path='D:/a/paper.pdf', start_page=3, end_page=5)",
    ],
    category="document",
    danger_level="safe",
)
def read_pdf_text(
    file_path: str,
    start_page: int = 1,
    end_page: int = 0,
    max_chars: int = 50000,
) -> Dict[str, Any]:
    """读取 PDF 文本。"""
    err = _check_path(file_path)
    if err:
        return err

    try:
        from pypdf import PdfReader
    except ImportError:
        return {"success": False, "error": "需要安装 pypdf: pip install pypdf"}

    try:
        reader = PdfReader(file_path)
        total = len(reader.pages)

        if start_page < 1:
            start_page = 1
        if end_page <= 0 or end_page > total:
            end_page = total
        if end_page < start_page:
            return {"success": False, "error": f"页码范围无效: {start_page}-{end_page}"}

        parts: List[str] = []
        char_count = 0
        pages_read = 0
        truncated = False

        for page_no in range(start_page, end_page + 1):
            page = reader.pages[page_no - 1]
            text = page.extract_text() or ""
            block = f"\n\n--- 第 {page_no} 页 ---\n{text}"

            if char_count + len(block) > max_chars:
                remaining = max_chars - char_count
                if remaining > 0:
                    parts.append(block[:remaining])
                    char_count = max_chars
                truncated = True
                break

            parts.append(block)
            char_count += len(block)
            pages_read += 1

        content = "".join(parts).strip()

        return {
            "success": True,
            "file_path": os.path.abspath(file_path),
            "total_pages": total,
            "pages_read": pages_read,
            "start_page": start_page,
            "end_page": start_page + pages_read - 1,
            "content": content,
            "char_count": char_count,
            "truncated": truncated,
            "has_more": (start_page + pages_read - 1) < total,
            "next_start_page": (start_page + pages_read) if truncated else None,
            "message": (
                f"读取 {pages_read} 页（共 {total} 页），"
                f"{char_count} 字符"
                + ("，已截断" if truncated else "")
            ),
        }
    except Exception as e:
        return {"success": False, "error": f"PDF 读取失败: {e}"}


@tool(
    name="read_pdf_metadata",
    description="读取 PDF 元数据（页数、标题、作者、创建时间）。",
    parameters={
        "file_path": {"type": "str", "description": "PDF 文件路径（必填）"},
    },
    examples=["read_pdf_metadata(file_path='D:/a/paper.pdf')"],
    category="document",
    danger_level="safe",
)
def read_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """读取 PDF 元数据。"""
    err = _check_path(file_path)
    if err:
        return err

    try:
        from pypdf import PdfReader
    except ImportError:
        return {"success": False, "error": "需要安装 pypdf"}

    try:
        reader = PdfReader(file_path)
        meta = reader.metadata or {}
        return {
            "success": True,
            "file_path": os.path.abspath(file_path),
            "total_pages": len(reader.pages),
            "title": meta.get("/Title"),
            "author": meta.get("/Author"),
            "subject": meta.get("/Subject"),
            "creator": meta.get("/Creator"),
            "producer": meta.get("/Producer"),
            "encrypted": reader.is_encrypted,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="merge_pdfs",
    description="合并多个 PDF 为一个文件。",
    parameters={
        "pdf_paths": {"type": "list", "description": "PDF 路径列表（按顺序合并）"},
        "output_path": {"type": "str", "description": "输出路径（必填）"},
    },
    examples=[
        "merge_pdfs(pdf_paths=['a.pdf', 'b.pdf'], output_path='merged.pdf')",
    ],
    category="document",
    danger_level="medium",
)
def merge_pdfs(pdf_paths: List[str], output_path: str) -> Dict[str, Any]:
    """合并 PDF。"""
    if not pdf_paths or len(pdf_paths) < 2:
        return {"success": False, "error": "至少需要 2 个 PDF 文件"}

    for p in pdf_paths:
        err = _check_path(p)
        if err:
            return err

    if not is_safe_path(os.path.abspath(output_path)):
        return {"success": False, "error": "输出路径受保护"}

    try:
        from pypdf import PdfWriter, PdfReader
    except ImportError:
        return {"success": False, "error": "需要安装 pypdf"}

    try:
        writer = PdfWriter()
        total_pages = 0
        for p in pdf_paths:
            reader = PdfReader(p)
            for page in reader.pages:
                writer.add_page(page)
                total_pages += 1

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            writer.write(f)

        return {
            "success": True,
            "output_path": os.path.abspath(output_path),
            "source_count": len(pdf_paths),
            "total_pages": total_pages,
            "message": f"已合并 {len(pdf_paths)} 个 PDF，共 {total_pages} 页 → {output_path}",
        }
    except Exception as e:
        return {"success": False, "error": f"合并失败: {e}"}


@tool(
    name="split_pdf",
    description="把 PDF 按页码范围拆分，或每 N 页拆成一个文件。",
    parameters={
        "file_path": {"type": "str", "description": "PDF 文件路径（必填）"},
        "output_dir": {"type": "str", "description": "输出目录（必填）"},
        "pages_per_file": {"type": "int", "description": "每个文件多少页，默认 1"},
    },
    examples=[
        "split_pdf(file_path='big.pdf', output_dir='D:/out', pages_per_file=1)",
    ],
    category="document",
    danger_level="medium",
)
def split_pdf(
    file_path: str,
    output_dir: str,
    pages_per_file: int = 1,
) -> Dict[str, Any]:
    """拆分 PDF。"""
    err = _check_path(file_path)
    if err:
        return err
    if pages_per_file < 1:
        return {"success": False, "error": "pages_per_file 必须 >= 1"}
    if not is_safe_path(os.path.abspath(output_dir)):
        return {"success": False, "error": "输出目录受保护"}

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return {"success": False, "error": "需要安装 pypdf"}

    try:
        reader = PdfReader(file_path)
        total = len(reader.pages)
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(file_path).stem

        outputs: List[str] = []
        for start in range(0, total, pages_per_file):
            end = min(start + pages_per_file, total)
            writer = PdfWriter()
            for i in range(start, end):
                writer.add_page(reader.pages[i])

            out_path = os.path.join(
                output_dir,
                f"{stem}_p{start + 1}-{end}.pdf",
            )
            with open(out_path, "wb") as f:
                writer.write(f)
            outputs.append(out_path)

        return {
            "success": True,
            "source": os.path.abspath(file_path),
            "total_pages": total,
            "output_files": outputs,
            "file_count": len(outputs),
            "message": f"已拆分为 {len(outputs)} 个文件，输出到 {output_dir}",
        }
    except Exception as e:
        return {"success": False, "error": f"拆分失败: {e}"}


# ═════════════════════════════════════════════════════════════
# Excel 写（不依赖 pandas，用 openpyxl）
# ═════════════════════════════════════════════════════════════

@tool(
    name="write_excel",
    description=(
        "从 list[dict] 数据生成 Excel 文件（.xlsx）。"
        "如果文件已存在则新建 sheet 或覆盖。"
    ),
    parameters={
        "data": {"type": "list", "description": "数据（list[dict] 或 list[list]）"},
        "file_path": {"type": "str", "description": "输出 .xlsx 路径（必填）"},
        "sheet_name": {"type": "str", "description": "sheet 名，默认 Sheet1"},
        "columns": {"type": "list", "description": "列顺序（可选，默认取第一行的键）"},
        "overwrite": {"type": "bool", "description": "已存在时是否覆盖，默认 True"},
    },
    examples=[
        "write_excel(data=[{'name':'Tom','age':20}], file_path='out.xlsx')",
    ],
    category="document",
    danger_level="medium",
)
def write_excel(
    data: List[Any],
    file_path: str,
    sheet_name: str = "Sheet1",
    columns: Optional[List[str]] = None,
    overwrite: bool = True,   # ← 默认覆盖
) -> Dict[str, Any]:
    """写 Excel。"""
    if not data:
        return {"success": False, "error": "数据为空"}

    if not is_safe_path(os.path.abspath(file_path)):
        return {"success": False, "error": "路径受保护"}

    if os.path.exists(file_path) and not overwrite:
        return {
            "success": False,
            "error": f"文件已存在: {file_path}（设 overwrite=True 覆盖）",
        }

    try:
        from openpyxl import Workbook
    except ImportError:
        return {"success": False, "error": "需要安装 openpyxl: pip install openpyxl"}

    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name

        # 表头
        if isinstance(data[0], dict):
            if columns is None:
                columns = list(data[0].keys())
            ws.append(columns)
            for row in data:
                ws.append([row.get(c) for c in columns])
        elif isinstance(data[0], (list, tuple)):
            for row in data:
                ws.append(list(row))
        else:
            return {"success": False, "error": "数据格式不支持（需 list[dict] 或 list[list]）"}

        wb.save(file_path)

        return {
            "success": True,
            "file_path": os.path.abspath(file_path),
            "sheet_name": sheet_name,
            "rows_written": len(data),
            "columns": columns,
            "message": f"已写入 {len(data)} 行到 {file_path}（sheet={sheet_name}）",
        }
    except Exception as e:
        return {"success": False, "error": f"写入失败: {e}"}


@tool(
    name="edit_excel",
    description=(
        "修改 Excel 里指定单元格的值。"
        "edits 格式：[{'sheet':'Sheet1','cell':'B2','value':100}]"
    ),
    parameters={
        "file_path": {"type": "str", "description": "Excel 路径（必填）"},
        "edits": {"type": "list", "description": "编辑列表（必填）"},
    },
    examples=[
        "edit_excel(file_path='a.xlsx', edits=[{'sheet':'Sheet1','cell':'B2','value':100}])",
    ],
    category="document",
    danger_level="medium",
)
def edit_excel(file_path: str, edits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """修改 Excel 单元格。"""
    err = _check_path(file_path)
    if err:
        return err
    if not edits:
        return {"success": False, "error": "edits 为空"}

    try:
        from openpyxl import load_workbook
    except ImportError:
        return {"success": False, "error": "需要安装 openpyxl"}

    try:
        wb = load_workbook(file_path)
        applied = []

        for edit in edits:
            sheet = edit.get("sheet")
            cell = edit.get("cell")
            value = edit.get("value")

            if not sheet or not cell:
                continue
            if sheet not in wb.sheetnames:
                return {"success": False, "error": f"Sheet 不存在: {sheet}"}

            ws = wb[sheet]
            ws[cell] = value
            applied.append({"sheet": sheet, "cell": cell, "value": value})

        wb.save(file_path)

        return {
            "success": True,
            "file_path": os.path.abspath(file_path),
            "edits_applied": len(applied),
            "details": applied,
            "message": f"已修改 {len(applied)} 个单元格",
        }
    except Exception as e:
        return {"success": False, "error": f"编辑失败: {e}"}


@tool(
    name="add_excel_sheet",
    description="往已有 Excel 里新增一个 sheet。",
    parameters={
        "file_path": {"type": "str", "description": "Excel 路径（必填）"},
        "sheet_name": {"type": "str", "description": "新 sheet 名（必填）"},
        "data": {"type": "list", "description": "数据（list[dict] 或 list[list]）"},
    },
    examples=[
        "add_excel_sheet(file_path='a.xlsx', sheet_name='Q2', data=[{'x':1}])",
    ],
    category="document",
    danger_level="medium",
)
def add_excel_sheet(
    file_path: str,
    sheet_name: str,
    data: List[Any],
) -> Dict[str, Any]:
    """加 sheet。"""
    err = _check_path(file_path)
    if err:
        return err

    try:
        from openpyxl import load_workbook
    except ImportError:
        return {"success": False, "error": "需要安装 openpyxl"}

    try:
        wb = load_workbook(file_path)
        if sheet_name in wb.sheetnames:
            return {"success": False, "error": f"Sheet 已存在: {sheet_name}"}

        ws = wb.create_sheet(sheet_name)
        if data and isinstance(data[0], dict):
            cols = list(data[0].keys())
            ws.append(cols)
            for row in data:
                ws.append([row.get(c) for c in cols])
        elif data and isinstance(data[0], (list, tuple)):
            for row in data:
                ws.append(list(row))

        wb.save(file_path)

        return {
            "success": True,
            "file_path": os.path.abspath(file_path),
            "sheet_name": sheet_name,
            "rows_written": len(data),
            "all_sheets": wb.sheetnames,
            "message": f"已新增 sheet '{sheet_name}'，写入 {len(data)} 行",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═════════════════════════════════════════════════════════════
# DOCX
# ═════════════════════════════════════════════════════════════

@tool(
    name="read_docx_text",
    description="读取 Word 文档（.docx）的段落文本。",
    parameters={
        "file_path": {"type": "str", "description": "Word 路径（必填）"},
        "max_chars": {"type": "int", "description": "最大字符数，默认 50000"},
    },
    examples=["read_docx_text(file_path='D:/a/report.docx')"],
    category="document",
    danger_level="safe",
)
def read_docx_text(file_path: str, max_chars: int = 50000) -> Dict[str, Any]:
    """读 Word 文本。"""
    err = _check_path(file_path)
    if err:
        return err

    try:
        from docx import Document
    except ImportError:
        return {"success": False, "error": "需要安装 python-docx"}

    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs]
        content = "\n".join(paragraphs)
        truncated = len(content) > max_chars

        return {
            "success": True,
            "file_path": os.path.abspath(file_path),
            "paragraph_count": len(paragraphs),
            "table_count": len(doc.tables),
            "content": content[:max_chars],
            "char_count": len(content),
            "truncated": truncated,
            "message": f"读取 {len(paragraphs)} 段落、{len(doc.tables)} 表格",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="read_docx_tables",
    description="读取 Word 文档里的所有表格（返回二维数组）。",
    parameters={
        "file_path": {"type": "str", "description": "Word 路径（必填）"},
    },
    examples=["read_docx_tables(file_path='D:/a/report.docx')"],
    category="document",
    danger_level="safe",
)
def read_docx_tables(file_path: str) -> Dict[str, Any]:
    """读 Word 表格。"""
    err = _check_path(file_path)
    if err:
        return err

    try:
        from docx import Document
    except ImportError:
        return {"success": False, "error": "需要安装 python-docx"}

    try:
        doc = Document(file_path)
        tables = []
        for ti, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                rows.append([cell.text for cell in row.cells])
            tables.append({"index": ti, "rows": rows, "row_count": len(rows)})

        return {
            "success": True,
            "file_path": os.path.abspath(file_path),
            "table_count": len(tables),
            "tables": tables,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="write_docx",
    description="从段落列表生成 Word 文档（.docx）。",
    parameters={
        "paragraphs": {"type": "list", "description": "段落列表（字符串列表）"},
        "file_path": {"type": "str", "description": "输出路径（必填）"},
        "title": {"type": "str", "description": "文档标题（可选）"},
    },
    examples=[
        "write_docx(paragraphs=['第一段', '第二段'], file_path='out.docx', title='报告')",
    ],
    category="document",
    danger_level="medium",
)
def write_docx(
    paragraphs: List[str],
    file_path: str,
    title: str = None,
) -> Dict[str, Any]:
    """写 Word。"""
    if not is_safe_path(os.path.abspath(file_path)):
        return {"success": False, "error": "路径受保护"}

    try:
        from docx import Document
    except ImportError:
        return {"success": False, "error": "需要安装 python-docx"}

    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        doc = Document()

        if title:
            doc.add_heading(title, level=1)

        for para in paragraphs:
            doc.add_paragraph(para)

        doc.save(file_path)

        return {
            "success": True,
            "file_path": os.path.abspath(file_path),
            "paragraph_count": len(paragraphs),
            "has_title": bool(title),
            "message": f"已写入 {len(paragraphs)} 段落到 {file_path}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="edit_docx",
    description="在 Word 文档里做查找替换。",
    parameters={
        "file_path": {"type": "str", "description": "Word 路径（必填）"},
        "find": {"type": "str", "description": "要查找的文本（必填）"},
        "replace": {"type": "str", "description": "替换为（必填）"},
    },
    examples=[
        "edit_docx(file_path='a.docx', find='旧', replace='新')",
    ],
    category="document",
    danger_level="medium",
)
def edit_docx(file_path: str, find: str, replace: str) -> Dict[str, Any]:
    """编辑 Word（查找替换）。"""
    err = _check_path(file_path)
    if err:
        return err

    try:
        from docx import Document
    except ImportError:
        return {"success": False, "error": "需要安装 python-docx"}

    try:
        doc = Document(file_path)
        replaced_count = 0

        for para in doc.paragraphs:
            for run in para.runs:
                if find in run.text:
                    run.text = run.text.replace(find, replace)
                    replaced_count += 1

        # 表格内也替换
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            if find in run.text:
                                run.text = run.text.replace(find, replace)
                                replaced_count += 1

        doc.save(file_path)

        return {
            "success": True,
            "file_path": os.path.abspath(file_path),
            "replaced_count": replaced_count,
            "message": (
                f"已替换 {replaced_count} 处"
                if replaced_count
                else "未找到匹配文本"
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}