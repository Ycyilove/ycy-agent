"""Persistent MCP client sessions and dynamic ToolAgent registration."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import MCPServerConfig

logger = logging.getLogger(__name__)


@dataclass
class MCPConnection:
    stack: AsyncExitStack
    session: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class MCPToolSpec:
    public_name: str
    server_name: str
    remote_name: str
    description: str
    input_schema: dict
    read_only_hint: bool = False
    idempotent_hint: bool = False


class MCPClientManager:
    """Connect to configured servers and expose their tools to ToolAgent."""

    def __init__(self, configs: List[MCPServerConfig]):
        self.configs = {config.name: config for config in configs}
        self.connections: Dict[str, MCPConnection] = {}
        self.tools: Dict[str, MCPToolSpec] = {}
        self.server_tool_names: Dict[str, set[str]] = {}
        self.resources: Dict[str, list] = {}
        self.resource_templates: Dict[str, list] = {}
        self.tried_servers: set[str] = set()
        self.statuses: Dict[str, dict] = {
            config.name: {
                "name": config.name,
                "transport": config.transport,
                "status": "configured",
                "tools": 0,
                "resources": 0,
            }
            for config in configs
        }

    def register_config(self, config: MCPServerConfig) -> None:
        """运行时注册一个未预先配置的 MCP 服务器。幂等。"""
        if config.name in self.configs:
            return
        self.configs[config.name] = config
        self.statuses[config.name] = {
            "name": config.name,
            "transport": config.transport,
            "status": "configured",
            "tools": 0,
            "resources": 0,
        }

    def unregister_config(self, server_name: str) -> None:
        """移除运行时注册的 MCP 服务器。调用方需先 disconnect。"""
        self.configs.pop(server_name, None)
        self.statuses.pop(server_name, None)
        self.connections.pop(server_name, None)
        self.tools = {
            k: v for k, v in self.tools.items() if v.server_name != server_name
        }
        self.server_tool_names.pop(server_name, None)
        self.resources.pop(server_name, None)
        self.resource_templates.pop(server_name, None)

    def has_server(self, server_name: str) -> bool:
        return server_name in self.configs

    @staticmethod
    def _headers(config: MCPServerConfig) -> dict[str, str]:
        return {
            header: os.environ[environment_name]
            for header, environment_name in config.header_env.items()
            if os.getenv(environment_name)
        }

    @staticmethod
    def _stdio_env(config: MCPServerConfig) -> dict[str, str]:
        return {
            key: os.environ[environment_name]
            for key, environment_name in config.env_from.items()
            if os.getenv(environment_name)
        }

    async def _connect_once(self, config: MCPServerConfig) -> MCPConnection:
        logger.info(
            "[dsh][mcp] _connect_once start: name=%s transport=%s url=%s",
            config.name, config.transport, config.url,
        )

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.sse import sse_client
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client
            from httpx2 import AsyncClient
        except ImportError as error:
            raise RuntimeError("MCP 客户端未安装，请执行 pip install mcp") from error

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            if config.transport == "stdio":
                parameters = StdioServerParameters(
                    command=config.command,
                    args=list(config.args),
                    env={**os.environ, **self._stdio_env(config)},
                )
                logger.info("[dsh][mcp] entering stdio_client: %s", config.name)
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(parameters)
                )
                logger.info("[dsh][mcp] stdio_client ready: %s", config.name)
            elif config.transport == "sse":
                logger.info("[dsh][mcp] entering sse_client: %s", config.url)
                read_stream, write_stream = await stack.enter_async_context(
                    sse_client(
                        config.url,
                        headers=self._headers(config),
                        timeout=config.connect_timeout,
                    )
                )
                logger.info("[dsh][mcp] sse_client ready: %s", config.name)
            else:
                logger.info("[dsh][mcp] creating AsyncClient: %s", config.url)
                http_client = await stack.enter_async_context(
                    AsyncClient(
                        headers=self._headers(config),
                        timeout=config.connect_timeout,
                    )
                )
                logger.info(
                    "[dsh][mcp] entering streamable_http_client: %s", config.url,
                )
                streams = await stack.enter_async_context(
                    streamable_http_client(
                        config.url,
                        http_client=http_client,
                    )
                )
                logger.info(
                    "[dsh][mcp] streamable_http_client ready: %s", config.name,
                )
                if isinstance(streams, tuple) and len(streams) >= 2:
                    read_stream, write_stream = streams[0], streams[1]
                else:
                    raise RuntimeError(
                        f"streamable_http_client 返回值异常: {streams!r}"
                    )

            logger.info("[dsh][mcp] entering ClientSession: %s", config.name)
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            logger.info(
                "[dsh][mcp] calling session.initialize(): %s", config.name,
            )
            await session.initialize()
            logger.info(
                "[dsh][mcp] session.initialize() done: %s", config.name,
            )
            return MCPConnection(stack=stack, session=session)

        except asyncio.CancelledError as error:
            logger.warning(
                "[dsh][mcp] _connect_once cancelled: %s (%s)",
                config.name, error,
            )
            try:
                await stack.aclose()
            except BaseException:
                pass
            raise RuntimeError(
                f"MCP 连接被内部取消（通常是连接失败）：{error}"
            ) from error

        except BaseException:
            try:
                await stack.aclose()
            except BaseException:
                pass
            raise

    async def connect(self, server_name: str) -> None:
        config = self.configs[server_name]
        last_error: Optional[BaseException] = None

        # 每次尝试的硬超时。比 config.connect_timeout 宽松 10 秒，
        # 防止 mcp SDK 内部某些步骤（DNS/TLS/task group）不受 AsyncClient timeout 约束。
        per_attempt_timeout = max(config.connect_timeout + 10, 25.0)

        for attempt in range(config.retries + 1):
            try:
                connection = await asyncio.wait_for(
                    self._connect_once(config),
                    timeout=per_attempt_timeout,
                )
                self.connections[server_name] = connection
                self.statuses[server_name].update(
                    {"status": "connected", "error": None}
                )
                return

            except asyncio.TimeoutError:
                last_error = RuntimeError(
                    f"MCP 连接整体超时 ({per_attempt_timeout}s) "
                    f"于第 {attempt + 1} 次尝试"
                )
                logger.warning(
                    "[dsh][mcp] connect %s attempt %d/%d hard timeout after %.1fs",
                    server_name, attempt + 1, config.retries + 1,
                    per_attempt_timeout,
                )
                if attempt < config.retries:
                    await asyncio.sleep(2 ** attempt)

            except (OSError, TimeoutError, ConnectionError, RuntimeError) as error:
                last_error = error
                logger.warning(
                    "[dsh][mcp] connect %s attempt %d/%d failed: %s: %s",
                    server_name, attempt + 1, config.retries + 1,
                    type(error).__name__, error,
                )
                if attempt < config.retries:
                    await asyncio.sleep(2 ** attempt)

        self.statuses[server_name].update(
            {"status": "error", "error": str(last_error)}
        )
        assert last_error is not None
        raise last_error

    async def discover(
        self,
        server_name: str,
        register_tool: Callable[..., Any],
        unregister_tool: Optional[Callable[[str], Any]] = None,
    ) -> None:
        connection = self.connections[server_name]
        config = self.configs[server_name]

        async with connection.lock:
            result = await asyncio.wait_for(
                connection.session.list_tools(),
                timeout=config.call_timeout,
            )

        server_tools = []
        if unregister_tool:
            for old_name in self.server_tool_names.get(server_name, set()):
                unregister_tool(old_name)
                self.tools.pop(old_name, None)

        for tool in getattr(result, "tools", []):
            raw = tool.model_dump(by_alias=True)
            annotations = raw.get("annotations") or {}
            public_name = f"mcp__{server_name}__{tool.name}"
            spec = MCPToolSpec(
                public_name=public_name,
                server_name=server_name,
                remote_name=tool.name,
                description=tool.description or "MCP 外部工具",
                input_schema=raw.get("inputSchema", {}),
                read_only_hint=bool(annotations.get("readOnlyHint", False)),
                idempotent_hint=bool(annotations.get("idempotentHint", False)),
            )
            self.tools[public_name] = spec
            server_tools.append(spec)

            properties = spec.input_schema.get("properties", {})

            # 用闭包包装 handler，确保空参数调用也能正确传递
            def _make_handler(pub_name: str):
                async def _handler(**kwargs):
                    return await self.call_tool(pub_name, kwargs)
                return _handler

            register_tool(
                name=spec.public_name,
                description=f"[MCP:{server_name}] {spec.description}",
                parameters=properties,
                input_schema=spec.input_schema,
                handler=_make_handler(spec.public_name),
                origin="mcp",
                danger_level="safe" if spec.read_only_hint else "high",
                idempotent=spec.idempotent_hint,
            )

        self.server_tool_names[server_name] = {
            spec.public_name for spec in server_tools
        }
        self.statuses[server_name]["tools"] = len(server_tools)

        try:
            async with connection.lock:
                resource_result = await asyncio.wait_for(
                    connection.session.list_resources(),
                    timeout=config.call_timeout,
                )
            self.resources[server_name] = [
                item.model_dump(by_alias=True)
                for item in getattr(resource_result, "resources", [])
            ]
        except Exception as error:
            logger.warning("MCP resource discovery failed for %s: %s", server_name, error)
            self.resources[server_name] = []

        try:
            async with connection.lock:
                template_result = await asyncio.wait_for(
                    connection.session.list_resource_templates(),
                    timeout=config.call_timeout,
                )
            self.resource_templates[server_name] = [
                item.model_dump(by_alias=True)
                for item in getattr(template_result, "resourceTemplates", [])
            ]
        except Exception as error:
            logger.warning(
                "MCP resource template discovery failed for %s: %s",
                server_name,
                error,
            )
            self.resource_templates[server_name] = []

        self.statuses[server_name]["resources"] = (
            len(self.resources[server_name])
            + len(self.resource_templates[server_name])
        )

    @staticmethod
    def _split_public_name(public_name: str) -> tuple[str, str, str]:
        """把 mcp__<server>__<tool> 拆成 (prefix, server_name, remote_name)。

        支持 server_name 里含 "-" 等非下划线字符。
        使用 split("__", 2) 保证只按前两个 __ 分割，
        这样 remote_name 里即使有 __ 也不会被误切。
        """
        if not isinstance(public_name, str) or not public_name.startswith("mcp__"):
            raise ValueError(f"不是合法的 MCP 工具名: {public_name!r}")
        parts = public_name.split("__", 2)
        if len(parts) != 3:
            raise ValueError(f"MCP 工具名格式错误: {public_name!r}")
        return parts[0], parts[1], parts[2]

    async def call_tool(self, public_name: str, arguments: dict):
        """调用 MCP 工具。

        关键设计：
        - 不依赖 self.tools[public_name]（会 KeyError）
          因为 MCP 服务器可能返回未在 tools/list 里列出的工具名
          （例如 A2AWire 的 confirm_keys_persisted）。
        - 直接从 public_name 解析 server_name 和 remote_name。
        - spec 变成可选，仅用于拿 idempotent_hint。
        """
        _, server_name, remote_name = self._split_public_name(public_name)

        config = self.configs.get(server_name)
        if config is None:
            raise RuntimeError(
                f"未配置的 MCP 服务: {server_name} (来自 {public_name})"
            )

        connection = self.connections.get(server_name)
        if connection is None:
            raise RuntimeError(
                f"MCP 服务未连接: {server_name} (来自 {public_name})"
            )

        # spec 可选：在 tools/list 里就取 idempotent_hint，否则默认 False
        spec = self.tools.get(public_name)
        idempotent_hint = spec.idempotent_hint if spec else False

        if spec is None:
            logger.info(
                "[dsh][mcp] on-demand tool call: %s "
                "(not in tools/list; server=%s, remote=%s)",
                public_name, server_name, remote_name,
            )

        last_error = None
        for attempt in range(config.retries + 1):
            try:
                async with connection.lock:
                    result = await asyncio.wait_for(
                        connection.session.call_tool(
                            remote_name,
                            arguments or {},
                        ),
                        timeout=config.call_timeout,
                    )
                return self._normalize_mcp_result(result.model_dump(by_alias=True))
            except (OSError, TimeoutError, ConnectionError) as error:
                last_error = error
                if not idempotent_hint or attempt >= config.retries:
                    raise
                await asyncio.sleep(2 ** attempt)

        # 理论不可达（循环里已经 raise），仅为静态检查
        raise last_error or RuntimeError("MCP call_tool failed")

    @staticmethod
    def _normalize_mcp_result(payload: dict) -> dict:
        """把 MCP 的标准返回结构标准化，让 ToolAgent 能正确判断 success。

        MCP 返回形如：
            {
                "content": [{"type": "text", "text": "..."}],
                "structuredContent": {...},
                "isError": False,
                "resultType": "complete"
            }

        当 isError=true，或 structuredContent.message_detail 说明缺少认证/参数时，
        把 success 设为 false，让上层能识别这是"未完成"状态。

        为什么需要这个转换：
        - MCP 用 HTTP 200 + isError 字段表示业务失败，和 ToolAgent 的
          success 字段语义不同
        - A2AWire 这类引导式服务器既返回 message_detail 又返回 next_action，
          这种情况应该保留 success=true，让模型继续推进
        """
        if not isinstance(payload, dict):
            return payload

        if "success" in payload:
            return payload

        is_error = bool(payload.get("isError"))
        structured = payload.get("structuredContent")
        message_detail = ""
        if isinstance(structured, dict):
            message_detail = str(structured.get("message_detail") or "")

        auth_keywords = (
            "missing", "required", "unauthenticated", "unauthorized",
            "not registered", "invalid", "expired", "forbidden",
        )
        detail_lower = message_detail.lower()
        needs_auth = any(kw in detail_lower for kw in auth_keywords)

        has_next_action = (
            isinstance(structured, dict)
            and isinstance(structured.get("next_action"), dict)
        )

        # 从 content 里提取错误文本作为兜底：
        # 很多 MCP 服务器不填 structuredContent.message_detail，
        # 而是把真实错误写在 content[0].text 里。
        # 不提取会导致错误信息丢失，模型无法判断问题原因。
        content_text = ""
        content_blocks = payload.get("content") or []
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text") or ""
                    if t:
                        content_text = t
                        break

        if is_error:
            payload["success"] = False
            payload["error"] = (
                message_detail
                or content_text
                or "MCP 工具返回错误（isError=true）"
            )
        elif needs_auth and not has_next_action:
            payload["success"] = False
            payload["error"] = message_detail or content_text
        else:
            payload["success"] = True

        return payload

    async def read_resource(self, server_name: str, uri: str):
        connection = self.connections.get(server_name)
        if connection is None:
            raise RuntimeError(f"MCP 服务未连接: {server_name}")
        config = self.configs[server_name]
        async with connection.lock:
            result = await asyncio.wait_for(
                connection.session.read_resource(uri),
                timeout=config.call_timeout,
            )
        return result.model_dump(by_alias=True)

    async def refresh(
        self,
        server_name: str,
        register_tool: Callable[..., Any],
        unregister_tool: Optional[Callable[[str], Any]] = None,
    ):
        await self.disconnect(server_name)
        await self.connect(server_name)
        await self.discover(server_name, register_tool, unregister_tool)

    async def start(
        self,
        register_tool: Callable[..., Any],
        unregister_tool: Optional[Callable[[str], Any]] = None,
    ) -> None:
        for server_name, config in self.configs.items():
            if not getattr(config, "enabled", True):
                logger.info("[dsh][mcp] skip disabled server: %s", server_name)
                self.statuses[server_name].update(
                    {"status": "disabled", "enabled": False}
                )
                continue
            try:
                await self.connect(server_name)
                await self.discover(server_name, register_tool, unregister_tool)
            except Exception as error:
                logger.exception("MCP server unavailable: %s", server_name)
                if server_name in self.connections:
                    await self.disconnect(server_name)
                self.statuses[server_name].update(
                    {"status": "error", "error": str(error)}
                )

    async def disconnect(self, server_name: str) -> None:
        connection = self.connections.pop(server_name, None)
        if connection is not None:
            await connection.stack.aclose()
        self.statuses.get(server_name, {}).update({"status": "disconnected"})

    async def close(self) -> None:
        for server_name in list(self.connections):
            await self.disconnect(server_name)

    def status(self) -> List[dict]:
        return list(self.statuses.values())

    def get_tools_by_servers(self, server_names: set) -> List[MCPToolSpec]:
        """根据服务器名称集合过滤工具"""
        filtered_tools = []
        for tool_spec in self.tools.values():
            if tool_spec.server_name in server_names:
                filtered_tools.append(tool_spec)
        return filtered_tools