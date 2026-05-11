"""
文件重命名/格式转换工具
提供文件操作、格式转换等功能，包含安全限制
"""
import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from . import tool, get_registry


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

        return {
            "success": True,
            "directory": directory,
            "files_count": len(files),
            "dirs_count": len(dirs),
            "files": sorted(files_info, key=lambda x: x['name']),
            "directories": sorted(dirs_info, key=lambda x: x['name'])
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
