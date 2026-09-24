"""
工具模块加载器
自动导入所有工具并注册到工具注册中心
"""

from . import get_registry
from .tools_def.sandbox import run_python_code
from .tools_def.data import read_csv, analyze_csv, read_excel, export_to_csv
from .tools_def.file import (
    list_files,
    rename_file,
    convert_file_format,
    get_file_info,
    read_text_file,
    write_file,
    edit_file,
    create_directory,
    move_file,
)
from .tools_def.time import (
    calculate_time_difference,
    add_time,
    get_current_time,
    format_timestamp,
    calculate_age
)
from .tools_def.network import (
    http_get,
    http_post,
    http_put,
    http_delete,
    check_url_status,
    parse_html,
    fetch_json
)
from .tools_def.mysql import (
    mysql_connect,
    mysql_query,
    mysql_execute,
    mysql_show_tables,
    mysql_describe_table,
    mysql_show_databases,
    mysql_count
)
from .tools_def import knowledge as knowledge_tools  # noqa: F401
from . import mcp_discovery  # noqa: F401


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
    'mysql_connect',
    'mysql_query',
    'mysql_execute',
    'mysql_show_tables',
    'mysql_describe_table',
    'mysql_show_databases',
    'mysql_count',
    'load_all_tools',
]