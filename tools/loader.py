"""
工具模块加载器
自动导入所有工具并注册到工具注册中心
"""

from . import get_registry
from .sandbox import run_python_code
from .data_tools import read_csv, analyze_csv, read_excel, export_to_csv
from .file_tools import list_files, rename_file, convert_file_format, get_file_info
from .time_tools import (
    calculate_time_difference,
    add_time,
    get_current_time,
    format_timestamp,
    calculate_age
)
from .network_tools import (
    http_get,
    http_post,
    http_put,
    http_delete,
    check_url_status,
    parse_html,
    fetch_json
)


def load_all_tools():
    """加载所有工具（触发装饰器执行）"""
    registry = get_registry()
    return registry


__all__ = [
    'get_registry',
    'run_python_code',
    'read_csv',
    'analyze_csv',
    'read_excel',
    'export_to_csv',
    'list_files',
    'rename_file',
    'convert_file_format',
    'get_file_info',
    'calculate_time_difference',
    'add_time',
    'get_current_time',
    'format_timestamp',
    'calculate_age',
    'http_get',
    'http_post',
    'http_put',
    'http_delete',
    'check_url_status',
    'parse_html',
    'fetch_json',
    'load_all_tools'
]
