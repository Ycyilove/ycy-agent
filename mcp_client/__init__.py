"""MCP client integration for the existing ToolAgent registry."""

from .config import MCPServerConfig, load_mcp_config
from .manager import MCPClientManager, MCPToolSpec

__all__ = [
    "MCPClientManager",
    "MCPServerConfig",
    "MCPToolSpec",
    "load_mcp_config",
]
