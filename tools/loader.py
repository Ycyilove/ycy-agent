"""
工具模块加载器
自动导入所有工具并注册到工具注册中心
"""

import logging

from tools import get_registry

_logger = logging.getLogger(__name__)


# ── 条件注册 run_python_code（仅 E2B 可用时） ──
def _register_sandbox_if_available() -> None:
    try:
        from .tools_def.sandbox import is_backend_available, run_python_code  # noqa: F401
    except Exception:
        _logger.exception("[loader] sandbox 模块加载失败，跳过 run_python_code")
        return
    if is_backend_available():
        _logger.info("[loader] run_python_code 已注册（E2B 后端可用）")
    else:
        # 装饰器已注册，需显式 unregister
        get_registry().unregister("run_python_code")
        _logger.warning(
            "[loader] run_python_code 未注册：E2B 后端不可用"
            "（SANDBOX_BACKEND!=e2b 或 E2B_API_KEY 缺失）"
        )


_register_sandbox_if_available()


# ── 其余内置工具（正常注册） ──
from .tools_def.data import read_csv, analyze_csv, read_excel, export_to_csv  # noqa: E402,F401
from .tools_def.file import (  # noqa: E402,F401
    list_files,
    rename_file,
    convert_file_format,
    get_file_info,
    read_text_file,
    write_file,
    edit_file,
    create_directory,
    move_file,
    delete_file,
    restore_file,
)
from .tools_def.file_search import (  # noqa: E402,F401
    search_content,
    search_files,
    read_file_range,
)
from .tools_def import documents  # noqa: E402,F401
from .tools_def.time import (  # noqa: E402,F401
    calculate_time_difference,
    add_time,
    get_current_time,
    format_timestamp,
    calculate_age,
)
from .tools_def.network import (  # noqa: E402,F401
    http_get,
    http_post,
    http_put,
    http_delete,
    check_url_status,
    parse_html,
    fetch_json,
)
from .tools_def.fetch import fetch_url  # noqa: E402,F401
from .tools_def.mysql import (  # noqa: E402,F401
    mysql_connect,
    mysql_query,
    mysql_execute,
    mysql_show_tables,
    mysql_describe_table,
    mysql_show_databases,
    mysql_count,
)
from .tools_def import knowledge as knowledge_tools  # noqa: E402,F401
from .tools_def.tool_result_reader import read_tool_result  # noqa: E402,F401
from . import mcp_discovery  # noqa: E402,F401
from tools.skills.loader import reload_skills as _reload_skills  # noqa: E402


def load_all_tools():
    registry = get_registry()
    try:
        _reload_skills()
    except Exception:
        _logger.exception("[skills] 加载失败")
    return registry


__all__ = [
    'get_registry',
    'read_csv',
    'analyze_csv',
    'read_excel',
    'export_to_csv',
    'list_files',
    'rename_file',
    'convert_file_format',
    'get_file_info',
    'create_directory',
    'move_file',
    'delete_file',
    'restore_file',
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
    'fetch_url',
    'read_tool_result',
    'mysql_connect',
    'mysql_query',
    'mysql_execute',
    'mysql_show_tables',
    'mysql_describe_table',
    'mysql_show_databases',
    'mysql_count',
    'load_all_tools',
]