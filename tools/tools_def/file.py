"""
文件重命名/格式转换工具
提供文件操作、格式转换等功能，包含安全限制
"""
import os
import shutil
import hashlib
import json
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .. import tool, get_registry


ALLOWED_EXTENSIONS = {
    'text': ['txt', 'md', 'json', 'xml', 'csv', 'yaml', 'yml', 'log', 'ini', 'cfg'],
    'code': ['py', 'js', 'ts', 'html', 'css', 'java', 'c', 'cpp', 'h', 'go', 'rs', 'rb', 'php'],
    'data': ['csv', 'json', 'xml', 'xlsx', 'xls', 'pdf'],
    'image': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg', 'ico'],
    'document': ['pdf', 'doc', 'docx', 'txt', 'md', 'rtf'],
}

PROTECTED_PATTERNS = [
    r'^C:\\Windows',
    r'^C:\\Program Files',
    r'^C:\\Program Files \(x86\)',
    r'^/etc',
    r'^/usr/bin',
    r'^/usr/lib',
    r'^/system',
]

# ─────────────────────────────────────────────────────────────
# 回收站（trash）
# ─────────────────────────────────────────────────────────────

_TRASH_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", ".trash",
    )
)
_TRASH_INDEX = os.path.join(_TRASH_DIR, "_index.json")
_trash_lock = threading.Lock()


def _ensure_trash_dir() -> None:
    os.makedirs(_TRASH_DIR, exist_ok=True)


def _load_trash_index() -> Dict[str, Any]:
    if not os.path.exists(_TRASH_INDEX):
        return {}
    try:
        with open(_TRASH_INDEX, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_trash_index(index: Dict[str, Any]) -> None:
    _ensure_trash_dir()
    tmp = _TRASH_INDEX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _TRASH_INDEX)


def move_to_trash(file_path: str) -> Dict[str, Any]:
    """把文件移到回收站。返回 {success, trash_id, ...}。"""
    if not os.path.exists(file_path):
        return {"success": False, "error": f"文件不存在: {file_path}"}
    if not is_safe_path(os.path.abspath(file_path)):
        return {"success": False, "error": "路径受保护"}

    try:
        _ensure_trash_dir()
        original_path = os.path.abspath(file_path)
        original_name = os.path.basename(original_path)
        trash_id = uuid.uuid4().hex[:8]
        trash_name = f"{trash_id}_{original_name}"
        trash_path = os.path.join(_TRASH_DIR, trash_name)

        size = os.path.getsize(original_path)
        shutil.move(original_path, trash_path)

        entry = {
            "id": trash_id,
            "original_path": original_path,
            "original_name": original_name,
            "trash_path": trash_path,
            "deleted_at": datetime.now().isoformat(),
            "size": size,
        }
        with _trash_lock:
            index = _load_trash_index()
            index[trash_id] = entry
            _save_trash_index(index)

        return {"success": True, "entry": entry}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_trash() -> List[Dict[str, Any]]:
    """列出回收站内容（按删除时间倒序）。"""
    with _trash_lock:
        index = _load_trash_index()
    items = [v for v in index.values() if os.path.exists(v.get("trash_path", ""))]
    items.sort(key=lambda x: x.get("deleted_at", ""), reverse=True)
    return items


def restore_from_trash(trash_id: str) -> Dict[str, Any]:
    """从回收站恢复文件。"""
    with _trash_lock:
        index = _load_trash_index()
        entry = index.get(trash_id)
        if not entry:
            return {"success": False, "error": f"回收站中不存在: {trash_id}"}

        trash_path = entry.get("trash_path")
        original_path = entry.get("original_path")

        if not os.path.exists(trash_path):
            index.pop(trash_id, None)
            _save_trash_index(index)
            return {"success": False, "error": "回收站中的文件已丢失"}

        # ── 原路径被占用 → 自动加时间戳后缀 ──
        final_path = original_path
        path_adjusted = False
        if os.path.exists(original_path):
            base, ext = os.path.splitext(original_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = f"{base}.restored_{timestamp}{ext}"
            path_adjusted = True
            # 极端情况：加后缀还冲突 → 再加短 uuid
            if os.path.exists(final_path):
                final_path = (
                    f"{base}.restored_{timestamp}_{uuid.uuid4().hex[:4]}{ext}"
                )

        try:
            os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
            shutil.move(trash_path, final_path)
            index.pop(trash_id, None)
            _save_trash_index(index)

            if path_adjusted:
                return {
                    "success": True,
                    "restored_path": final_path,
                    "original_path": original_path,
                    "path_adjusted": True,
                    "message": (
                        f"原路径 {original_path} 已被占用，"
                        f"文件已恢复到 {final_path}"
                    ),
                }
            return {
                "success": True,
                "restored_path": final_path,
                "message": f"已恢复到 {final_path}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def purge_from_trash(trash_id: str) -> Dict[str, Any]:
    """从回收站永久删除。"""
    with _trash_lock:
        index = _load_trash_index()
        entry = index.get(trash_id)
        if not entry:
            return {"success": False, "error": f"回收站中不存在: {trash_id}"}

        trash_path = entry.get("trash_path")
        try:
            if os.path.exists(trash_path):
                if os.path.isdir(trash_path):
                    shutil.rmtree(trash_path)
                else:
                    os.remove(trash_path)
            index.pop(trash_id, None)
            _save_trash_index(index)
            return {"success": True, "message": "已永久删除", "path": entry.get("original_path", ""),}
        except Exception as e:
            return {"success": False, "error": str(e)}


def purge_all_trash() -> Dict[str, Any]:
    """清空回收站。"""
    with _trash_lock:
        index = _load_trash_index()
        count = 0
        for entry in index.values():
            trash_path = entry.get("trash_path")
            try:
                if os.path.exists(trash_path):
                    if os.path.isdir(trash_path):
                        shutil.rmtree(trash_path)
                    else:
                        os.remove(trash_path)
                    count += 1
            except Exception:
                pass
        _save_trash_index({})
        return {"success": True, "count": count}
        
def is_protected_path(path: str) -> bool:
    """检查是否为受保护路径"""
    path = os.path.abspath(path)
    for pattern in PROTECTED_PATTERNS:
        import re
        if re.match(pattern, path, re.IGNORECASE):
            return True
    return False


def is_safe_path(path: str, base_dir: str = None) -> bool:
    """检查路径是否安全（在允许的目录内）"""
    abs_path = os.path.abspath(path)

    if is_protected_path(abs_path):
        return False

    if base_dir:
        abs_base = os.path.abspath(base_dir)
        return abs_path.startswith(abs_base)

    return True


@tool(
    name="list_files",
    description="列出指定目录下的文件，支持过滤和排序",
    parameters={
        "directory": {"type": "str", "description": "要列出的目录路径"},
        "pattern": {"type": "str", "description": "文件过滤模式（可选）"},
        "include_subdirs": {"type": "bool", "description": "是否包含子目录，默认False"}
    },
    examples=[
        "list_files(directory='.')",
        "list_files(directory='./data', pattern='*.csv')"
    ],
    category="file",
    danger_level="safe"
)
def list_files(directory: str, pattern: str = None, include_subdirs: bool = False) -> Dict[str, Any]:
    """列出目录文件"""
    try:
        if not os.path.exists(directory):
            return {"success": False, "error": f"目录不存在: {directory}"}

        if not os.path.isdir(directory):
            return {"success": False, "error": f"不是有效目录: {directory}"}

        files = []
        dirs = []

        if include_subdirs:
            for root, dirs_list, files_list in os.walk(directory):
                for f in files_list:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, directory)
                    if pattern and not Path(f).match(pattern):
                        continue
                    files.append(rel_path)
                for d in dirs_list:
                    full_path = os.path.join(root, d)
                    rel_path = os.path.relpath(full_path, directory)
                    dirs.append(rel_path)
        else:
            for item in os.listdir(directory):
                full_path = os.path.join(directory, item)
                if os.path.isfile(full_path):
                    if pattern and not Path(item).match(pattern):
                        continue
                    files.append(item)
                elif os.path.isdir(full_path):
                    dirs.append(item)

        files_info = []
        for f in files:
            full_path = os.path.join(directory, f)
            stat = os.stat(full_path)
            files_info.append({
                "name": f,
                "size": stat.st_size,
                "size_display": format_file_size(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })

        dirs_info = []
        for d in dirs:
            full_path = os.path.join(directory, d)
            stat = os.stat(full_path)
            dirs_info.append({
                "name": d,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })

        abs_dir = os.path.abspath(directory)
        return {
            "success": True,
            "directory": abs_dir,
            "files_count": len(files),
            "dirs_count": len(dirs),
            "files": sorted(files_info, key=lambda x: x['name']),
            "directories": sorted(dirs_info, key=lambda x: x['name']),
            "message": (
                f"已列出 {abs_dir}："
                f"{len(files)} 个文件、{len(dirs)} 个子目录"
                + (f"（过滤：{pattern}）" if pattern else "")
            ),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def format_file_size(size: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


@tool(
    name="rename_file",
    description="重命名文件或移动文件到新位置",
    parameters={
        "old_path": {"type": "str", "description": "原文件路径"},
        "new_name": {"type": "str", "description": "新文件名（不含路径）或新路径"}
    },
    examples=[
        "rename_file(old_path='old.txt', new_name='new.txt')",
        "rename_file(old_path='./data/file.csv', new_name='./backup/file.csv')"
    ],
    category="file",
    danger_level="medium"
)
def rename_file(old_path: str, new_name: str) -> Dict[str, Any]:
    """重命名或移动文件"""
    try:
        if not os.path.exists(old_path):
            return {"success": False, "error": f"文件不存在: {old_path}"}

        old_abs = os.path.abspath(old_path)
        if not is_safe_path(old_abs):
            return {"success": False, "error": "操作被拒绝：路径受保护"}

        if os.path.sep in new_name or os.path.isabs(new_name):
            new_path = new_name
        else:
            new_path = os.path.join(os.path.dirname(old_path), new_name)

        new_abs = os.path.abspath(new_path)
        if not is_safe_path(new_abs, os.path.dirname(old_abs)):
            return {"success": False, "error": "操作被拒绝：目标路径受保护"}

        if os.path.exists(new_path):
            return {"success": False, "error": f"目标文件已存在: {new_path}"}

        os.makedirs(os.path.dirname(new_path) or '.', exist_ok=True)
        os.rename(old_path, new_path)

        return {
            "success": True,
            "message": f"文件已重命名",
            "old_path": old_path,
            "new_path": new_path
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="convert_file_format",
    description="文件格式转换，支持文本编码转换、JSON格式化等",
    parameters={
        "file_path": {"type": "str", "description": "源文件路径"},
        "target_format": {"type": "str", "description": "目标格式（如 txt, json, csv）"},
        "output_path": {"type": "str", "description": "输出文件路径（可选）"}
    },
    examples=[
        "convert_file_format(file_path='data.txt', target_format='json')",
        "convert_file_format(file_path='data.csv', target_format='json', output_path='data.json')"
    ],
    category="file",
    danger_level="medium"
)
def convert_file_format(file_path: str, target_format: str, output_path: str = None) -> Dict[str, Any]:
    """文件格式转换"""
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"文件不存在: {file_path}"}

        if not is_safe_path(os.path.abspath(file_path)):
            return {"success": False, "error": "操作被拒绝：路径受保护"}

        source_ext = Path(file_path).suffix.lstrip('.').lower()
        target_ext = target_format.lower().lstrip('.')

        if output_path:
            output_abs = os.path.abspath(output_path)
            if not is_safe_path(output_abs):
                return {"success": False, "error": "操作被拒绝：输出路径受保护"}
        else:
            output_path = str(Path(file_path).with_suffix('.' + target_ext))

        if source_ext == 'txt' and target_ext == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            with open(output_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(lines, f, ensure_ascii=False, indent=2)
            return {"success": True, "message": f"已转换为JSON格式", "output": output_path}

        elif source_ext == 'json' and target_ext == 'txt':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, (list, dict)):
                content = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                content = str(data)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"success": True, "message": f"已转换为文本格式", "output": output_path}

        elif source_ext == 'csv' and target_ext == 'json':
            import json
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return {"success": True, "message": f"已转换为JSON格式", "output": output_path}

        elif source_ext == 'json' and target_ext == 'csv':
            import json
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not data:
                return {"success": False, "error": "JSON文件为空"}
            if isinstance(data, dict):
                data = [data]
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            return {"success": True, "message": f"已转换为CSV格式", "output": output_path}

        elif source_ext == target_ext:
            shutil.copy2(file_path, output_path)
            return {"success": True, "message": f"文件已复制", "output": output_path}

        else:
            return {"success": False, "error": f"不支持的转换: {source_ext} -> {target_ext}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="get_file_info",
    description="获取文件详细信息（大小、创建时间、修改时间、哈希值等）",
    parameters={
        "file_path": {"type": "str", "description": "文件路径"},
        "compute_hash": {"type": "bool", "description": "是否计算哈希值，默认False"}
    },
    examples=[
        "get_file_info(file_path='data.csv')",
        "get_file_info(file_path='data.csv', compute_hash=True)"
    ],
    category="file",
    danger_level="safe"
)
def get_file_info(file_path: str, compute_hash: bool = False) -> Dict[str, Any]:
    """获取文件信息"""
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"文件不存在: {file_path}"}

        if not os.path.isfile(file_path):
            return {"success": False, "error": f"不是文件: {file_path}"}

        stat = os.stat(file_path)
        abs_path = os.path.abspath(file_path)

        info = {
            "success": True,
            "name": os.path.basename(file_path),
            "path": abs_path,
            "directory": os.path.dirname(abs_path),
            "extension": Path(file_path).suffix.lstrip('.'),
            "size": stat.st_size,
            "size_display": format_file_size(stat.st_size),
            "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "accessed": datetime.fromtimestamp(stat.st_atime).strftime("%Y-%m-%d %H:%M:%S"),
        }

        if compute_hash:
            with open(file_path, 'rb') as f:
                info["md5"] = hashlib.md5(f.read()).hexdigest()
            with open(file_path, 'rb') as f:
                info["sha256"] = hashlib.sha256(f.read()).hexdigest()

        return info

    except Exception as e:
        return {"success": False, "error": str(e)}

# 禁止用 write_file 写的二进制扩展名 → 提示改用哪个工具
_BINARY_WRITE_HINTS = {
    ".xlsx": "write_excel",
    ".xls":  "write_excel",
    ".docx": "write_docx",
    ".doc":  "write_docx",
    ".pdf":  "merge_pdfs 或专用工具",
    ".pptx": "（当前无写入工具）",
    ".zip":  "（当前无写入工具）",
}


@tool(
    name="write_file",
    description="创建新文件或覆盖写入现有文件。用于创建文本/代码/配置文件。",
    parameters={
        "file_path": {"type": "str", "description": "文件路径（必填）"},
        "content": {"type": "str", "description": "要写入的文本内容（必填）"},
        "encoding": {"type": "str", "description": "编码，默认 utf-8"},
    },
    examples=[
        "write_file(file_path='D:/data/notes.txt', content='hello')",
    ],
    category="file",
    danger_level="medium",
)
def write_file(file_path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """创建或覆盖文件。"""
    try:
        # ── 二进制格式保护：拒绝写 .xlsx/.docx/.pdf 等 ──
        ext = os.path.splitext(file_path)[1].lower()
        if ext in _BINARY_WRITE_HINTS:
            hint = _BINARY_WRITE_HINTS[ext]
            return {
                "success": False,
                "error": (
                    f"❌ 不能用 write_file 写 {ext} 文件（二进制格式）。"
                    f"请改用工具：{hint}"
                ),
                "hint_tool": hint,
            }

        if not is_safe_path(os.path.abspath(file_path)):
            return {"success": False, "error": "操作被拒绝：路径受保护"}
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
        abs_path = os.path.abspath(file_path)
        return {
            "success": True,
            "message": (
                f"已写入 {abs_path}"
                f"（{len(content)} 字符，{len(content.encode(encoding))} 字节，"
                f"{content.count(chr(10)) + (1 if content else 0)} 行）"
            ),
            "file_path": abs_path,
            "size": len(content),
            "bytes": len(content.encode(encoding)),
            "line_count": content.count("\n") + (1 if content else 0),
            "encoding": encoding,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool(
    name="edit_file",
    description="在文件里做局部替换。用于修改现有文件内容（非全量覆盖）。",
    parameters={
        "file_path": {"type": "str", "description": "文件路径（必填）"},
        "old_text": {"type": "str", "description": "要替换的旧文本（必填）"},
        "new_text": {"type": "str", "description": "替换后的新文本（必填）"},
    },
    examples=[
        "edit_file(file_path='notes.txt', old_text='test', new_text='production')",
    ],
    category="file",
    danger_level="medium",
)
def edit_file(file_path: str, old_text: str, new_text: str) -> Dict[str, Any]:
    """局部替换文件内容。"""
    try:
        if not is_safe_path(os.path.abspath(file_path)):
            return {"success": False, "error": "操作被拒绝：路径受保护"}
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            preview = old_text[:80] + ("..." if len(old_text) > 80 else "")
            return {
                "success": False,
                "error": (
                    f"文件中未找到要替换的内容。"
                    f"你传入的 old_text（前 80 字符）：{preview!r}\n"
                    f"提示：old_text 必须逐字匹配文件内容（含空格、缩进、换行）。"
                ),
                "file_path": os.path.abspath(file_path),
                "old_text_preview": preview,
            }
        new_content = content.replace(old_text, new_text, 1)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        abs_path = os.path.abspath(file_path)
        delta_lines = (
            new_content.count("\n") - content.count("\n")
        )
        return {
            "success": True,
            "message": (
                f"已替换 {abs_path}"
                f"（old_text {len(old_text)} 字符 → new_text {len(new_text)} 字符"
                + (
                    f"，行数 {delta_lines:+d}"
                    if delta_lines != 0
                    else ""
                )
                + "）"
            ),
            "file_path": abs_path,
            "old_text_preview": old_text[:80],
            "new_text_preview": new_text[:80],
            "old_size": len(content),
            "new_size": len(new_content),
            "delta_lines": delta_lines,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="create_directory",
    description="创建目录（含父目录）。用于新建文件夹。",
    parameters={
        "directory": {"type": "str", "description": "目录路径（必填）"},
    },
    examples=["create_directory(directory='D:/data/new_folder')"],
    category="file",
    danger_level="safe",
)
def create_directory(directory: str) -> Dict[str, Any]:
    """创建目录。"""
    try:
        abs_dir = os.path.abspath(directory)
        if not is_safe_path(abs_dir):
            return {"success": False, "error": "操作被拒绝：路径受保护"}
        already_existed = os.path.exists(abs_dir)
        os.makedirs(abs_dir, exist_ok=True)
        return {
            "success": True,
            "message": (
                f"目录已{'存在' if already_existed else '创建'}：{abs_dir}"
            ),
            "directory": abs_dir,
            "already_existed": already_existed,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="move_file",
    description="移动或重命名文件。",
    parameters={
        "source": {"type": "str", "description": "源文件路径（必填）"},
        "destination": {"type": "str", "description": "目标路径（必填）"},
    },
    examples=["move_file(source='a.txt', destination='subdir/a.txt')"],
    category="file",
    danger_level="medium",
)
def move_file(source: str, destination: str) -> Dict[str, Any]:
    """移动/重命名。"""
    try:
        if not is_safe_path(os.path.abspath(source)):
            return {"success": False, "error": "源路径受保护"}
        if not is_safe_path(os.path.abspath(destination)):
            return {"success": False, "error": "目标路径受保护"}
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        abs_src = os.path.abspath(source)
        abs_dst = os.path.abspath(destination)
        shutil.move(abs_src, abs_dst)
        return {
            "success": True,
            "message": f"已移动：{abs_src} → {abs_dst}",
            "source": abs_src,
            "new_path": abs_dst,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="delete_file",
    description=(
        "删除文件（移到回收站，可恢复）。"
        "删除后文件暂存到 data/.trash/，不会立刻从磁盘消失。"
    ),
    parameters={
        "file_path": {"type": "str", "description": "要删除的文件路径（必填）"},
    },
    examples=["delete_file(file_path='D:/data/old.txt')"],
    category="file",
    danger_level="medium",
)
def delete_file(file_path: str) -> Dict[str, Any]:
    """删除文件到回收站。"""
    try:
        result = move_to_trash(file_path)
        if not result.get("success"):
            return result
        entry = result["entry"]
        return {
            "success": True,
            "message": f"已移入回收站：{entry['original_name']}",
            "original_path": entry["original_path"],
            "trash_id": entry["id"],
            "new_path": entry["trash_path"],   # 让 _file_log_entry 能取到路径
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="restore_file",
    description="从回收站恢复文件到原始路径。",
    parameters={
        "trash_id": {"type": "str", "description": "回收站条目的 ID（必填）"},
    },
    examples=["restore_file(trash_id='abc12345')"],
    category="file",
    danger_level="medium",
)
def restore_file(trash_id: str) -> Dict[str, Any]:
    """恢复回收站文件。"""
    return restore_from_trash(trash_id)

@tool(
    name="read_text_file",
    description="读取文本文件内容（与 read_csv 不同，不解析结构）。",
    parameters={
        "file_path": {"type": "str", "description": "文件路径（必填）"},
        "encoding": {"type": "str", "description": "编码，默认 utf-8"},
    },
    examples=["read_text_file(file_path='notes.txt')"],
    category="file",
    danger_level="safe",
)
def read_text_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """读文本。"""
    try:
        if not is_safe_path(os.path.abspath(file_path)):
            return {"success": False, "error": "路径受保护"}
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        truncated = len(content) > 50000
        return {
            "success": True,
            "content": content[:50000],
            "file_path": os.path.abspath(file_path),
            "size": len(content),
            "line_count": content.count("\n") + (1 if content else 0),
            "truncated": truncated,
            "message": (
                f"已读取 {os.path.abspath(file_path)}"
                f"（{len(content)} 字符，{content.count(chr(10)) + 1} 行"
                + ("，已截断前 50000 字符" if truncated else "")
                + "）"
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}