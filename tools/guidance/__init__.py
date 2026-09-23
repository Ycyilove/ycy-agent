"""Guidance extraction subpackage.

从任意 MCP 工具返回里提取"下一步引导"，分层设计：
    L1：MCP 协议层（isError / content[].text / structuredContent）
    L2：语义扫描（递归找"看起来像工具调用"的 dict）
    L3：路径规则（按路径判断 mandatory vs optional）
    L4：关键词调整（用 message / 工具名微调）

对外接口：
    extract_guidance(payload) -> dict
"""

from .extractor import extract_guidance

__all__ = ["extract_guidance"]