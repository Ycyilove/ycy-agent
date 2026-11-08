"""Configuration loading for external MCP servers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


Transport = Literal["stdio", "sse", "http"]
_SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: Transport
    command: Optional[str] = None
    args: tuple[str, ...] = ()
    url: Optional[str] = None
    env_from: Dict[str, str] = field(default_factory=dict)
    header_env: Dict[str, str] = field(default_factory=dict)
    # headers: 明文 header 值（如 {"Authorization": "Bearer sk-xxx"}）。
    # 与 header_env 的区别：header_env 是「header名 -> 本机环境变量名」，
    # headers 是「header名 -> 值本身」。两者会合并，headers 优先。
    headers: Dict[str, str] = field(default_factory=dict)
    connect_timeout: float = 10.0
    call_timeout: float = 30.0
    retries: int = 2
    enabled: bool = True
    allow_hidden_tools: tuple[str, ...] = ()
    capability_tags: tuple[str, ...] = () 

    @classmethod
    def from_mapping(cls, name: str, value: dict) -> "MCPServerConfig":
        if not _SERVER_NAME.fullmatch(name):
            raise ValueError(
                f"非法 MCP 服务名称: {name!r}，只允许字母、数字、下划线和连字符"
            )

        transport = value.get("transport")
        if transport not in {"stdio", "sse", "http"}:
            raise ValueError(f"MCP 服务 {name} 的 transport 无效: {transport!r}")

        command = value.get("command")
        url = value.get("url")
        if transport == "stdio" and not command:
            raise ValueError(f"stdio MCP 服务 {name} 必须配置 command")
        if transport in {"sse", "http"} and not url:
            raise ValueError(f"{transport} MCP 服务 {name} 必须配置 url")

        # env_from 语义：{子进程变量名: 本机环境变量名}
        # 生态方言 env 语义：{变量名: 明文值}
        # 为了兼容配置文件里直接写 env，这里把 env 的 key 转成「同名引用」，
        # 值不直接注入（否则会绕过白名单，且和 env_from 语义冲突）。
        env_from = dict(value.get("env_from", {}))
        raw_env = value.get("env")
        if isinstance(raw_env, dict):
            for key in raw_env.keys():
                env_from.setdefault(str(key), str(key))

        return cls(
            name=name,
            transport=transport,
            command=command,
            args=tuple(str(item) for item in value.get("args", [])),
            url=url,
            env_from=env_from,
            header_env=dict(value.get("header_env", {})),
            headers={
                str(k): str(v)
                for k, v in (value.get("headers") or {}).items()
            },
            connect_timeout=max(0.1, float(value.get("connect_timeout", 10))),
            call_timeout=max(0.1, float(value.get("call_timeout", 30))),
            retries=max(0, int(value.get("retries", 2))),
            enabled=bool(value.get("enabled", True)),
            allow_hidden_tools=tuple(
                str(t) for t in value.get("allow_hidden_tools", [])
            ),
            capability_tags=tuple(
                str(t) for t in value.get("capability_tags", [])
            ),
        )


def _server_mappings(raw: dict) -> dict:
    """从配置文件里取出 server 映射。

    兼容三种顶层结构：
      { "mcpServers": {...} }   —— Cursor / Claude Code / DSH 生态通用
      { "servers":    {...} }   —— 本项目原有方言
      { "mcp": { "servers": {...} } } —— 旧嵌套
    也兼容 servers 为 list 的形态（每项带 name）。
    """
    if isinstance(raw.get("mcp"), dict):
        raw = raw["mcp"]

    servers = raw.get("mcpServers")
    if not isinstance(servers, (dict, list)):
        servers = raw.get("servers", {})

    if isinstance(servers, list):
        return {
            item["name"]: {key: value for key, value in item.items() if key != "name"}
            for item in servers
            if isinstance(item, dict) and item.get("name")
        }
    return servers if isinstance(servers, dict) else {}


def load_mcp_config(path: Optional[str] = None) -> List[MCPServerConfig]:
    """Load optional JSON configuration without making startup mandatory."""
    config_path = (
        path
        or os.getenv("MCP_CONFIG_FILE")
        or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "mcp_servers.json",
        )
    )
    if not config_path or not os.path.exists(config_path):
        return []

    file_path = Path(config_path)
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    return [
        MCPServerConfig.from_mapping(name, value)
        for name, value in _server_mappings(raw).items()
    ]
