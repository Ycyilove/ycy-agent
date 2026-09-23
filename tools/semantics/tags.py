"""Tool intent tags.

策略 A：给每个工具打"用途标签"，让 LLM 按标签选工具而不是按名字猜。
"""

from __future__ import annotations

from typing import Dict, List, Optional


# 用途标签的中文描述（供 prompt 展示）
INTENT_TAG_DESCRIPTIONS: Dict[str, str] = {
    "read_text": "读取文本内容",
    "read_binary": "读取二进制内容",
    "write_new": "创建新文件",
    "write_modify": "修改现有文件内容",
    "read_dir": "列出目录内容",
    "traverse_dir": "递归遍历目录",
    "search_name": "按文件名搜索",
    "search_content": "按文件内容搜索",
    "move": "移动或重命名",
    "mkdir": "创建目录",
    "stat": "查看文件元信息",
    "convert": "转换文件格式",
    "execute_code": "执行代码",
    "network_read": "发送 HTTP 读取请求",
    "network_write": "发送 HTTP 写请求",
    "parse": "解析 HTML/XML",
    "db_connect": "连接数据库",
    "db_read": "数据库读操作",
    "db_write": "数据库写操作",
    "compute": "数值/时间计算",
    "knowledge_read": "知识库检索",
}


# 工具基名 → 用途标签
_TOOL_INTENT_TAGS_BY_SUFFIX: Dict[str, List[str]] = {
    # MCP filesystem
    "read_text_file": ["read_text"],
    "read_file": ["read_text", "read_binary"],
    "read_media_file": ["read_binary"],
    "read_multiple_files": ["read_text"],
    "write_file": ["write_new", "write_modify"],
    "edit_file": ["write_modify"],
    "create_directory": ["mkdir"],
    "list_directory": ["read_dir"],
    "list_directory_with_sizes": ["read_dir"],
    "directory_tree": ["traverse_dir"],
    "search_files": ["search_name", "search_content"],
    "move_file": ["move"],
    "get_file_info": ["stat"],
    "list_allowed_directories": ["read_dir"],

    # 内置文件
    "read_csv": ["read_text"],
    "analyze_csv": ["read_text"],
    "read_excel": ["read_text"],
    "export_to_csv": ["write_new"],
    "list_files": ["read_dir"],
    "rename_file": ["move"],
    "convert_file_format": ["convert"],

    # 沙箱
    "run_python_code": ["execute_code"],

    # 网络
    "http_get": ["network_read"],
    "http_post": ["network_write"],
    "http_put": ["network_write"],
    "http_delete": ["network_write"],
    "check_url_status": ["network_read"],
    "parse_html": ["parse"],
    "fetch_json": ["network_read"],

    # MySQL
    "mysql_connect": ["db_connect"],
    "mysql_query": ["db_read"],
    "mysql_execute": ["db_write"],
    "mysql_show_tables": ["db_read"],
    "mysql_describe_table": ["db_read"],
    "mysql_show_databases": ["db_read"],
    "mysql_count": ["db_read"],

    # 时间
    "calculate_time_difference": ["compute"],
    "add_time": ["compute"],
    "get_current_time": ["compute"],
    "format_timestamp": ["compute"],
    "calculate_age": ["compute"],

    # 知识库
    "search_knowledge": ["knowledge_read"],
}


def tool_basename(tool_name: str) -> str:
    """把 mcp__server__tool 拆成 tool；普通工具名原样返回。"""
    if not tool_name:
        return ""
    if "__" not in tool_name:
        return tool_name
    return tool_name.rsplit("__", 1)[-1]


def get_intent_tags(tool_name: str) -> List[str]:
    """获取某个工具的所有用途标签。"""
    basename = tool_basename(tool_name)
    return list(_TOOL_INTENT_TAGS_BY_SUFFIX.get(basename, []))


def get_intent_tag_description(tag: str) -> str:
    return INTENT_TAG_DESCRIPTIONS.get(tag, tag)


def find_tools_by_intent(
    intent: str,
    available_tools,
) -> List[str]:
    """从可用工具里找匹配某个意图的工具名列表。"""
    matched: List[str] = []
    for name in available_tools:
        tags = get_intent_tags(name)
        if intent in tags:
            matched.append(name)
    return matched