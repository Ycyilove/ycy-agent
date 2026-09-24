"""Client for the public MCP Registry.

Provides keyword search over published MCP servers and a helper to
convert a registry entry into a runtime MCPServerConfig.

设计原则：
    - 只接受 ASCII 关键词。AI 必须生成英文关键词，中文会被自动过滤。
    - 多关键词按优先级降级搜索。
    - 网络层有重试和分层超时，禁用代理环境变量干扰。
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

# 和 manager.py 保持一致：优先 httpx2
try:
    import httpx2 as httpx  # type: ignore
except ImportError:
    import httpx  # type: ignore

from .config import MCPServerConfig


logger = logging.getLogger(__name__)


REGISTRY_BASE = "https://registry.modelcontextprotocol.io"
REGISTRY_SEARCH_PATH = "/v0/servers"

# ── 网络参数 ──
_MAX_RETRIES = 3
_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=60.0,
    write=10.0,
    pool=10.0,
)
_USER_AGENT = "MCP-Client/1.0 (+https://modelcontextprotocol.io)"

# 查询长度上限，超过直接截断，避免 Registry 拒绝
_MAX_QUERY_LEN = 128


# 合法的 Registry 本地名：字母、数字、下划线、连字符
_SERVER_NAME_SAFE = re.compile(r"[^A-Za-z0-9_-]+")

# 只匹配 ASCII token（字母/数字/下划线/连字符）。
# 中文、空格、标点都会被过滤掉。
# 这样即使 AI 传了中文，也只搜英文字符（可能为空）。
_QUERY_TOKEN = re.compile(r"[A-Za-z0-9_\-]+")


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────


def _safe_local_name(registry_name: str) -> str:
    """把注册表里的 server name 转成符合本地配置规则的短名。"""
    short = registry_name.replace("/", "_").replace(".", "_")
    short = _SERVER_NAME_SAFE.sub("_", short)
    return short[:64] or "mcp_remote"


def _pick_remote_raw(server: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 registry 条目里挑出第一个可用的 remote，保留原始字段。"""
    remotes = server.get("remotes") or []
    if not isinstance(remotes, list):
        return None
    for remote in remotes:
        if not isinstance(remote, dict):
            continue
        url = remote.get("url")
        if not url:
            continue
        kind = remote.get("type") or "streamable-http"
        if kind in ("streamable-http", "http"):
            return {"type": "http", "url": url, "headers": remote.get("headers") or []}
        if kind == "sse":
            return {"type": "sse", "url": url, "headers": remote.get("headers") or []}
    return None


def _tokenize_query(query: str) -> List[str]:
    """把查询拆成 ASCII token。

    只保留字母/数字/下划线/连字符。
    中文、空格、标点会被丢弃——AI 必须传英文关键词。
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if len(query) > _MAX_QUERY_LEN:
        logger.warning(
            "[dsh][registry] query too long (%d chars), truncating to %d",
            len(query), _MAX_QUERY_LEN,
        )
        query = query[:_MAX_QUERY_LEN]
    tokens = _QUERY_TOKEN.findall(query)
    seen = set()
    result = []
    for t in tokens:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _dedupe_by_registry_name(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 server.name 去重。"""
    seen = set()
    result = []
    for entry in entries:
        server = entry.get("server") if "server" in entry else entry
        if not isinstance(server, dict):
            continue
        name = server.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(entry)
    return result


def _count_token_hits(entry: Dict[str, Any], tokens: List[str]) -> int:
    """统计命中 token 数（用于排序）。"""
    if not tokens:
        return 0
    server = entry.get("server") if "server" in entry else entry
    if not isinstance(server, dict):
        return 0
    haystack = " ".join(
        str(server.get(k) or "").lower()
        for k in ("name", "description", "title")
    )
    return sum(1 for t in tokens if t.lower() in haystack)


def _entries_to_configs(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把原始条目转成精简配置列表。"""
    results: List[Dict[str, Any]] = []
    for entry in entries:
        server = entry.get("server") if "server" in entry else entry
        if not isinstance(server, dict):
            continue
        remote = _pick_remote_raw(server)
        if not remote:
            continue
        registry_name = server.get("name") or ""
        results.append(
            {
                "registry_name": registry_name,
                "local_name": _safe_local_name(registry_name),
                "description": server.get("description") or "",
                "transport": remote["type"],
                "url": remote["url"],
                "headers": remote.get("headers") or [],
            }
        )
    return results


# ─────────────────────────────────────────────────────────────
# 底层搜索（含重试 + 禁用代理）
# ─────────────────────────────────────────────────────────────


async def _search_once(
    query: str,
    limit: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    """单次搜索，含重试。

    Returns:
        (entries, network_ok)
        - entries: 结果列表（可能为空）
        - network_ok: True = 网络层成功；False = 网络层失败
    """
    params = {"search": query, "limit": str(limit)}
    url = f"{REGISTRY_BASE}{REGISTRY_SEARCH_PATH}"
    headers = {"User-Agent": _USER_AGENT}

    last_error: Optional[BaseException] = None

    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT,
                headers=headers,
                follow_redirects=True,
                # 关键：忽略 HTTP_PROXY / HTTPS_PROXY 等环境变量。
                # 你的 curl.exe 能连，httpx 超时，最可能就是这个原因。
                trust_env=False,
            ) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()

            servers = payload.get("servers") or []
            if not isinstance(servers, list):
                servers = []

            logger.info(
                "[dsh][registry] search %r → %d entries (attempt %d)",
                query, len(servers), attempt + 1,
            )
            return servers, True

        except Exception as error:
            last_error = error
            logger.warning(
                "[dsh][registry] search %r attempt %d/%d failed: %s: %s",
                query, attempt + 1, _MAX_RETRIES,
                type(error).__name__, error or "(no message)",
            )
            if attempt < _MAX_RETRIES - 1:
                backoff = 2 ** attempt
                logger.info(
                    "[dsh][registry] retrying in %d seconds", backoff,
                )
                await asyncio.sleep(backoff)

    logger.error(
        "[dsh][registry] search %r failed after %d attempts: %s: %s",
        query, _MAX_RETRIES,
        type(last_error).__name__,
        last_error or "(no message)",
    )
    return [], False


# ─────────────────────────────────────────────────────────────
# 公开 API
# ─────────────────────────────────────────────────────────────


async def search_servers(
    query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """搜索 Registry，返回原始条目。

    策略：
        1. token 化（只保留 ASCII）
        2. 单 token：直接搜
        3. 多 token：先整体搜；无结果则逐 token 搜、合并、按命中数排序
    """
    tokens = _tokenize_query(query)
    if not tokens:
        logger.warning(
            "[dsh][registry] query %r has no ASCII token, "
            "AI must provide English keywords",
            query,
        )
        return []

    if len(tokens) == 1:
        entries, _ = await _search_once(tokens[0], limit)
        return entries

    # 多 token：先整体搜
    full_query = " ".join(tokens)
    primary, _ = await _search_once(full_query, limit)
    if primary:
        return primary

    # 逐 token 搜
    all_entries: List[Dict[str, Any]] = []
    for token in tokens:
        entries, _ = await _search_once(token, limit)
        all_entries.extend(entries)

    deduped = _dedupe_by_registry_name(all_entries)
    deduped.sort(
        key=lambda e: _count_token_hits(e, tokens),
        reverse=True,
    )
    return deduped


async def search_configs(
    query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """单关键词搜索，返回精简配置。"""
    entries = await search_servers(query, limit=limit)
    return _entries_to_configs(entries)


async def search_configs_multi(
    keywords: List[str],
    limit: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """按优先级依次尝试多个关键词。

    Returns:
        (configs, meta)
        meta = {
            "matched_keyword": str | None,
            "network_error": bool,
            "tried_keywords": list,
        }
    """
    tried: List[str] = []
    network_failures = 0

    for keyword in keywords:
        keyword = keyword.strip()
        if not keyword:
            continue

        tokens = _tokenize_query(keyword)
        if not tokens:
            logger.warning(
                "[dsh][registry] keyword %r has no ASCII token, skipping",
                keyword,
            )
            continue

        tried.append(keyword)

        # 单 token：直接搜
        if len(tokens) == 1:
            entries, network_ok = await _search_once(tokens[0], limit)
        else:
            # 多 token：先整体搜
            full_query = " ".join(tokens)
            entries, network_ok = await _search_once(full_query, limit)
            if not entries and network_ok:
                # 整体无结果但网络正常 → 降级逐 token 搜
                all_entries: List[Dict[str, Any]] = []
                token_network_ok = True
                for token in tokens:
                    sub, ok = await _search_once(token, limit)
                    all_entries.extend(sub)
                    if not ok:
                        token_network_ok = False
                entries = _dedupe_by_registry_name(all_entries)
                entries.sort(
                    key=lambda e: _count_token_hits(e, tokens),
                    reverse=True,
                )
                network_ok = token_network_ok

        if not network_ok:
            network_failures += 1
            logger.warning(
                "[dsh][registry] keyword %r: network error", keyword,
            )
            continue

        if entries:
            configs = _entries_to_configs(entries)
            if configs:
                logger.info(
                    "[dsh][registry] keyword %r → %d configs",
                    keyword, len(configs),
                )
                return configs, {
                    "matched_keyword": keyword,
                    "network_error": False,
                    "tried_keywords": tried,
                }

        logger.info(
            "[dsh][registry] keyword %r → 0 results, trying next",
            keyword,
        )

    all_network_errors = (bool(tried) and network_failures == len(tried))
    return [], {
        "matched_keyword": None,
        "network_error": all_network_errors,
        "tried_keywords": tried,
    }


async def config_for_entry(entry: Dict[str, Any]) -> Optional[MCPServerConfig]:
    """从精简配置构造 MCPServerConfig。"""
    if not entry.get("url") or not entry.get("transport"):
        return None

    local_name = entry.get("local_name") or _safe_local_name(
        entry.get("registry_name") or ""
    )

    header_env: Dict[str, str] = {}
    for header in entry.get("headers") or []:
        if not isinstance(header, dict):
            continue
        name = header.get("name")
        if not name or not header.get("isRequired"):
            continue
        env_name = "MCP_" + local_name.upper() + "_" + name.upper().replace("-", "_")
        header_env[name] = env_name

    try:
        return MCPServerConfig.from_mapping(
            local_name,
            {
                "transport": entry["transport"],
                "url": entry["url"],
                "header_env": header_env,
                "connect_timeout": 15,
                "call_timeout": 30,
                "retries": 1,
            },
        )
    except ValueError as error:
        logger.warning("MCP registry entry rejected: %s", error)
        return None