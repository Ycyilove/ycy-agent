"""文件搜索与分页读取工具。

补充 data.py 的缺口：
    - search_content : 按正则搜文件内容（类似 grep）
    - search_files   : 按 glob 搜文件名（类似 find）
    - read_file_range: 按行号分页读文件（避免大文件被截断）

设计约束：
    - 复用 data.py 的路径白名单思路（可选，如果 data.py 有）
    - 返回结构统一为 dict，方便 format_tool_result 处理
    - 单次返回有上限，避免 prompt 爆炸
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import tool


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────────────────────

# 单次搜索/读取的最大返回量
_MAX_MATCHES = 200
_MAX_FILES_SCANNED = 2000
_MAX_FILE_SIZE = 5 * 1024 * 1024   # 5MB，超过的文件跳过内容搜索

# 默认跳过目录
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".idea", ".vscode", "dist", "build", ".next", ".pytest_cache",
    "models",  # 你项目的模型缓存，通常很大
})

# 默认跳过的二进制扩展名
_BINARY_EXT = frozenset({
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".dat",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".mp3", ".mp4", ".avi", ".mov",
    ".pdf", ".docx", ".xlsx",
    ".db", ".sqlite", ".sqlite3",
})


def _iter_files(
    root: Path,
    name_glob: str = "*",
    skip_dirs: frozenset = _SKIP_DIRS,
):
    """递归遍历文件，跳过指定目录。"""
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地修改 dirnames 以跳过目录
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs and not d.startswith(".")
        ]

        for name in filenames:
            if not fnmatch.fnmatch(name, name_glob):
                continue
            scanned += 1
            if scanned > _MAX_FILES_SCANNED:
                logger.warning(
                    "[file_search] 扫描文件数超过 %d，提前停止", _MAX_FILES_SCANNED
                )
                return
            yield Path(dirpath) / name


def _is_text_file(path: Path) -> bool:
    """粗略判断是否为文本文件。"""
    if path.suffix.lower() in _BINARY_EXT:
        return False
    try:
        if path.stat().st_size > _MAX_FILE_SIZE:
            return False
    except OSError:
        return False
    return True


def _read_lines(path: Path) -> Optional[List[str]]:
    """读文件为行列表，编码失败返回 None。"""
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except (UnicodeDecodeError, LookupError):
            continue
        except OSError:
            return None
    return None


# ─────────────────────────────────────────────────────────────
# 工具 1：search_content —— 内容搜索（grep）
# ─────────────────────────────────────────────────────────────

@tool(
    name="search_content",
    description=(
        "在**本地文件系统**中递归搜索文件内容，支持正则表达式。"
        "返回匹配的文件、行号和该行内容。"
        "适用于「哪个文件定义了 xxx」「哪里用了 yyy」这类问题。"
        "⚠️ **只搜文件系统，搜不到知识库（RAG）。**"
        "如果要查用户导入的 PDF/DOCX/TXT 文档，请用 `search_knowledge`。"
    ),
    parameters={
        "pattern": {"type": "str", "description": "正则表达式或普通字符串"},
        "path": {"type": "str", "description": "搜索根目录，默认当前目录"},
        "file_glob": {"type": "str", "description": "文件名过滤，默认 *（所有文件）"},
        "case_sensitive": {"type": "bool", "description": "是否区分大小写，默认 False"},
        "is_regex": {"type": "bool", "description": "pattern 是否为正则，默认 True"},
        "max_results": {"type": "int", "description": "最多返回多少条匹配，默认 50"},
    },
    examples=[
        "search_content(pattern='def run_python_code', path='D:/ycy/LLM/LLM', file_glob='*.py')",
        "search_content(pattern='import os', path='./tools', is_regex=False)",
    ],
    category="file",
    danger_level="safe",
)
def search_content(
    pattern: str,
    path: str = ".",
    file_glob: str = "*",
    case_sensitive: bool = False,
    is_regex: bool = True,
    max_results: int = 50,
) -> Dict[str, Any]:
    """在文件内容里搜索匹配。"""
    root = Path(path).resolve()
    if not root.is_dir():
        return {
            "success": False,
            "error": f"路径不是目录: {root}",
        }

    # 编译正则
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        if is_regex:
            regex = re.compile(pattern, flags)
        else:
            regex = re.compile(re.escape(pattern), flags)
    except re.error as e:
        return {
            "success": False,
            "error": f"正则表达式错误: {e}",
        }

    cap = min(max_results, _MAX_MATCHES)
    matches: List[Dict[str, Any]] = []
    files_scanned = 0
    files_matched = 0

    for file_path in _iter_files(root, name_glob=file_glob):
        if not _is_text_file(file_path):
            continue

        lines = _read_lines(file_path)
        if lines is None:
            continue

        files_scanned += 1
        file_had_match = False

        for line_no, line in enumerate(lines, start=1):
            if regex.search(line):
                file_had_match = True
                matches.append({
                    "file": str(file_path),
                    "line": line_no,
                    "content": line.rstrip()[:300],
                })
                if len(matches) >= cap:
                    break

        if file_had_match:
            files_matched += 1

        if len(matches) >= cap:
            break

    return {
        "success": True,
        "pattern": pattern,
        "root": str(root),
        "matches": matches,
        "match_count": len(matches),
        "files_scanned": files_scanned,
        "files_matched": files_matched,
        "truncated": len(matches) >= cap,
        "message": (
            f"找到 {len(matches)} 处匹配，"
            f"涉及 {files_matched} 个文件"
            + ("（已达上限，结果可能不完整）" if len(matches) >= cap else "")
        ),
    }


# ─────────────────────────────────────────────────────────────
# 工具 2：search_files —— 文件名搜索（find）
# ─────────────────────────────────────────────────────────────

@tool(
    name="search_files",
    description=(
        "在目录下递归搜索文件名，支持 glob 模式（如 *.py、test_*.txt）。"
        "适用于「找一下所有的配置文件」「test 开头的文件在哪」这类问题。"
    ),
    parameters={
        "pattern": {"type": "str", "description": "glob 模式，如 *.py"},
        "path": {"type": "str", "description": "搜索根目录，默认当前目录"},
        "max_results": {"type": "int", "description": "最多返回多少个文件，默认 100"},
    },
    examples=[
        "search_files(pattern='*.py', path='D:/ycy/LLM/LLM/tools')",
        "search_files(pattern='test_*.py', path='.')",
    ],
    category="file",
    danger_level="safe",
)
def search_files(
    pattern: str,
    path: str = ".",
    max_results: int = 100,
) -> Dict[str, Any]:
    """按 glob 模式搜索文件名。"""
    root = Path(path).resolve()
    if not root.is_dir():
        return {
            "success": False,
            "error": f"路径不是目录: {root}",
        }

    cap = min(max_results, _MAX_MATCHES)
    found: List[Dict[str, Any]] = []

    for file_path in _iter_files(root, name_glob=pattern):
        try:
            stat = file_path.stat()
            size = stat.st_size
        except OSError:
            size = 0

        found.append({
            "path": str(file_path),
            "name": file_path.name,
            "size": size,
            "size_human": _human_size(size),
        })

        if len(found) >= cap:
            break

    return {
        "success": True,
        "pattern": pattern,
        "root": str(root),
        "files": found,
        "count": len(found),
        "truncated": len(found) >= cap,
        "message": f"找到 {len(found)} 个文件"
        + ("（已达上限）" if len(found) >= cap else ""),
    }


# ─────────────────────────────────────────────────────────────
# 工具 3：read_file_range —— 分页读取
# ─────────────────────────────────────────────────────────────

@tool(
    name="read_file_range",
    description=(
        "按行号范围读取文件，用于大文件分页阅读。"
        "返回指定行的内容 + 文件总行数 + 是否还有更多。"
        "当 read_text_file 返回被截断时，用这个工具继续读。"
    ),
    parameters={
        "file_path": {"type": "str", "description": "文件路径"},
        "start_line": {"type": "int", "description": "起始行号（从 1 开始），默认 1"},
        "end_line": {"type": "int", "description": "结束行号（含），默认 500"},
    },
    examples=[
        "read_file_range(file_path='D:/a/big.log', start_line=1, end_line=100)",
        "read_file_range(file_path='D:/a/big.log', start_line=101, end_line=200)",
    ],
    category="file",
    danger_level="safe",
)
def read_file_range(
    file_path: str,
    start_line: int = 1,
    end_line: int = 500,
) -> Dict[str, Any]:
    """读取文件的指定行范围。"""
    p = Path(file_path).resolve()
    if not p.is_file():
        return {
            "success": False,
            "error": f"文件不存在: {p}",
        }

    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line

    # 单次读取上限，防止一次拉太多
    MAX_SPAN = 2000
    if end_line - start_line + 1 > MAX_SPAN:
        end_line = start_line + MAX_SPAN - 1

    lines = _read_lines(p)
    if lines is None:
        return {
            "success": False,
            "error": f"无法读取文件（可能是二进制或编码问题）: {p}",
        }

    total = len(lines)
    # 转成 0-based 切片
    s = start_line - 1
    e = min(end_line, total)

    if s >= total:
        return {
            "success": True,
            "file_path": str(p),
            "start_line": start_line,
            "end_line": start_line - 1,
            "total_lines": total,
            "content": "",
            "has_more": False,
            "message": f"起始行 {start_line} 超过文件总行数 {total}",
        }

    selected = lines[s:e]
    content = "\n".join(selected)

    return {
        "success": True,
        "file_path": str(p),
        "start_line": start_line,
        "end_line": e,
        "total_lines": total,
        "content": content,
        "line_count": len(selected),
        "has_more": e < total,
        "next_start_line": e + 1 if e < total else None,
        "message": (
            f"读取第 {start_line}-{e} 行，共 {len(selected)} 行"
            f"（文件总行数 {total}）"
            + (f"，可用 start_line={e+1} 继续读" if e < total else "，已到文件末尾")
        ),
    }


# ─────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────

def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"