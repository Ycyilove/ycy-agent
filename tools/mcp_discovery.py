"""Discovery tools that let the orchestrator search the public MCP Registry."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

from . import tool


logger = logging.getLogger(__name__)


_manager_provider: Optional[Callable[[], Any]] = None
_register_tool: Optional[Callable[..., Any]] = None
_unregister_tool: Optional[Callable[[str], Any]] = None
_llm_provider: Optional[Callable[[], Any]] = None


def set_mcp_discovery_context(
    manager_provider: Callable[[], Any],
    register_tool: Callable[..., Any],
    unregister_tool: Optional[Callable[[str], Any]],
    llm_provider: Optional[Callable[[], Any]],
) -> None:
    global _manager_provider, _register_tool, _unregister_tool, _llm_provider
    _manager_provider = manager_provider
    _register_tool = register_tool
    _unregister_tool = unregister_tool
    _llm_provider = llm_provider


def _get_manager():
    if _manager_provider is None:
        raise RuntimeError("MCP discovery context is not configured")
    manager = _manager_provider()
    if manager is None:
        raise RuntimeError("MCP client manager is not available")
    return manager


def _get_tried_servers() -> Set[str]:
    try:
        manager = _get_manager()
    except Exception:
        return set()
    if not hasattr(manager, "tried_servers"):
        manager.tried_servers = set()
    return manager.tried_servers


def reset_tried_servers() -> None:
    try:
        manager = _get_manager()
    except Exception:
        return
    if not hasattr(manager, "tried_servers"):
        manager.tried_servers = set()
    manager.tried_servers.clear()


# ─────────────────────────────────────────────────────────────
# 启发式排序（不调 LLM）
# ─────────────────────────────────────────────────────────────

# 地理相关的关键词
_GEO_US_ONLY = (
    "nws", "us weather", "usa weather", "united states",
    "us-only", "us only", "noaa", "weather.gov",
)
_GEO_CHINA = ("china", "chinese", "中国", "cn.")
_GEO_GLOBAL = (
    "global", "worldwide", "any location", "open-meteo", "openmeteo",
    "multi-model", "weathermesh",
)


def _score_candidate(
    candidate: Dict[str, Any],
    capability: str,
    user_hint: str = "",
) -> int:
    """启发式打分，分越高越优先。

    评分维度：
        + 命中 capability token 数
        + 无认证
        + 描述里含"global" / "worldwide"
        + URL 简单（短）
        - 描述里含"US only"（如果不是美国用户）
        - 描述里含"require auth" / "api key"
    """
    score = 0

    desc = (candidate.get("description") or "").lower()
    name = (candidate.get("registry_name") or "").lower()
    url = (candidate.get("url") or "").lower()
    haystack = f"{desc} {name}"

    # capability token 命中数
    cap_tokens = [
        t.strip().lower() for t in capability.split(",")
        if t.strip()
    ]
    for t in cap_tokens:
        if t in haystack:
            score += 10

    # user_hint（例如用户问"广州"，hint 可以是 "china"）
    if user_hint and user_hint.lower() in haystack:
        score += 15

    # 无认证
    if not candidate.get("headers"):
        score += 5

    # 地理匹配
    if any(kw in haystack for kw in _GEO_CHINA):
        # 如果 user_hint 提到 china 相关，大幅加分
        if user_hint and "china" in user_hint.lower():
            score += 30
        else:
            score += 10
    if any(kw in haystack for kw in _GEO_GLOBAL):
        score += 8
    if any(kw in haystack for kw in _GEO_US_ONLY):
        # 如果是中国用户，大幅减分
        if user_hint and "china" in user_hint.lower():
            score -= 50
        else:
            score -= 5

    # URL 简单度（越短越优先，但不要过度）
    url_len = len(url)
    if url_len < 40:
        score += 3
    elif url_len > 80:
        score -= 2

    return score


def _heuristic_rank(
    capability: str,
    candidates: List[Dict[str, Any]],
    user_hint: str = "",
) -> List[Dict[str, Any]]:
    """按启发式分数排序候选，返回排序后的列表。"""
    scored = [
        (_score_candidate(c, capability, user_hint), i, c)
        for i, c in enumerate(candidates)
    ]
    # 分数高的优先，同分保持原序
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, _, c in scored]


# ─────────────────────────────────────────────────────────────
# LLM 排序（带超时 + 预筛选）
# ─────────────────────────────────────────────────────────────

# 给 LLM 排序的候选上限（避免 prompt 过长）
_LLM_RANK_TOP_N = 10
# LLM 排序超时（秒）
_LLM_RANK_TIMEOUT = 15.0


async def _rank_candidates(
    capability: str,
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """从候选里挑最相关的一个。

    策略：
        1. 启发式排序
        2. 如果 capability 里有明确的地理词（china/us）→ 直接用启发式 Top 1
        3. 如果候选数 <= 5 → 直接用启发式 Top 1
        4. 否则才调 LLM 排序（带 8s 超时）
    """
    if not candidates:
        return None

    # 提取地理提示
    cap_lower = capability.lower()
    user_hint = ""
    if "china" in cap_lower or "chinese" in cap_lower:
        user_hint = "china"
    elif "us" in cap_lower or "usa" in cap_lower:
        user_hint = "us"

    ranked = _heuristic_rank(capability, candidates, user_hint)

    # ── 决策：是否需要 LLM ──
    # 规则 1：候选 <= 5，启发式够
    if len(ranked) <= 5:
        logger.info(
            "[dsh][discovery] %d candidates, using heuristic top 1: %s",
            len(ranked), ranked[0]["registry_name"],
        )
        return ranked[0]

    # 规则 2：capability 有明确地理词 → 启发式够
    if user_hint:
        logger.info(
            "[dsh][discovery] geo hint=%r, using heuristic top 1: %s",
            user_hint, ranked[0]["registry_name"],
        )
        return ranked[0]

    # 规则 3：启发式 Top 1 分数明显高于 Top 2 → 启发式够
    if len(ranked) >= 2:
        top1_score = _score_candidate(ranked[0], capability, user_hint)
        top2_score = _score_candidate(ranked[1], capability, user_hint)
        if top1_score - top2_score >= 5:
            logger.info(
                "[dsh][discovery] heuristic margin %d, using top 1: %s",
                top1_score - top2_score, ranked[0]["registry_name"],
            )
            return ranked[0]

    # ── 以上规则都没命中 → 调 LLM ──
    if _llm_provider is None:
        return ranked[0]

    top_n = ranked[:_LLM_RANK_TOP_N]

    lines = []
    for i, c in enumerate(top_n, 1):
        needs_auth = "需要认证" if c.get("headers") else "无需认证"
        lines.append(
            f"{i}. registry_name={c['registry_name']}\n"
            f"   description={c['description'][:200]}\n"
            f"   认证要求={needs_auth}"
        )

    prompt = (
        "用户能力需求：" + capability + "\n\n"
        "候选 MCP 服务器（已按启发式预排序）：\n" + "\n".join(lines) + "\n\n"
        "请按以下优先级选择最匹配的服务器：\n\n"
        "**优先级 1：地理覆盖必须匹配用户需求。**\n"
        "   - 用户问具体地区时，先看 description 里的地理范围。\n"
        "   - 含 'US' / 'USA' / 'United States' / 'NWS' 的**只覆盖美国**。\n"
        "   - 含 'China' / 'Chinese city' 的**优先给中国用户**。\n"
        "   - 含 'global' / 'worldwide' / 'any location' / 'Open-Meteo' "
        "的**全球通用**。\n\n"
        "**优先级 2：功能匹配。**\n"
        "**优先级 3：无需认证优先。**\n\n"
        "只返回最相关的那一个服务器的序号（纯数字）。\n"
        "如果都不相关，返回 0。"
    )

    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                _llm_provider().generate,
                [{"role": "user", "content": prompt}],
            ),
            timeout=_LLM_RANK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[dsh][discovery] LLM ranking timed out after %.1fs, "
            "falling back to heuristic top 1: %s",
            _LLM_RANK_TIMEOUT, top_n[0]["registry_name"],
        )
        return top_n[0]
    except Exception:
        logger.exception(
            "[dsh][discovery] LLM ranking failed, "
            "falling back to heuristic top 1"
        )
        return top_n[0]

    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    if not digits:
        return top_n[0]
    idx = int(digits)
    if idx <= 0 or idx > len(top_n):
        # LLM 说"都不相关"，但启发式可能有分
        # 如果启发式 Top 1 分数 > 0，还是返回它
        if _score_candidate(top_n[0], capability, user_hint) > 0:
            return top_n[0]
        return None
    return top_n[idx - 1]


def _format_tools_for_message(tools: List[str]) -> str:
    if not tools:
        return "（该服务器没有暴露任何工具）"
    lines = ["**以下是你下一步必须使用到的确切工具名**（不要自己构造）："]
    for t in tools:
        lines.append(f"  - {t}")
    lines.append("")
    lines.append(
        "**调用时请使用上面的完整工具名**，格式为 mcp__<server>__<tool>。"
        "**绝对不要简化工具名**。"
    )
    return "\n".join(lines)


@tool(
    name="search_and_connect_mcp",
    description=(
        "Search the public MCP Registry for a server that provides the given "
        "capability, connect to it, and return the tools it exposes. "
        "Use this when the user needs external data or actions you cannot do "
        "with existing tools.\n\n"
        "**The capability parameter MUST contain only English keywords**, "
        "separated by commas, ordered from most specific to most general. "
        "For example: 'weather-forecast,weather,forecast'. "
        "The system will try each keyword in order until it finds results. "
        "**NEVER pass Chinese text.**\n\n"
        "**IMPORTANT: Each call picks a DIFFERENT candidate.** "
        "Servers already connected in this session are excluded from future "
        "searches. If the currently connected server does NOT match the "
        "user's needs (wrong geographic coverage, missing features, etc.), "
        "**call this tool again with a broader keyword** to get a different "
        "server.\n\n"
        "**After this tool returns, use the exact tool names from the "
        "`tools` list. Do NOT construct or guess tool names.**"
    ),
    parameters={
        "capability": {
            "type": "str",
            "description": (
                "能力关键词，**只允许英文**，用逗号分隔多个候选。"
                "示例：'weather-forecast,weather,forecast'。"
                "**禁止传中文**。"
                "系统会依次尝试每个关键词。"
                "**每次调用会排除已连接过的服务器**——"
                "如果需要换一个服务器，用更宽的关键词再调一次。"
            ),
        },
        "limit": {
            "type": "int",
            "description": "每个关键词的搜索候选条数上限，默认 100",
        },
    },
    examples=[
        "search_and_connect_mcp(capability='weather-forecast,weather,forecast')",
        "search_and_connect_mcp(capability='search')",
        "search_and_connect_mcp(capability='filesystem,file')",
    ],
    category="mcp",
    danger_level="safe",
)
async def search_and_connect_mcp(
    capability: str,
    limit: int = 100,
) -> Dict[str, Any]:
    """Search registry, rank, connect, return exposed tools."""
    from mcp_client.registry_client import (
        search_configs_multi,
        config_for_entry,
    )

    keywords = [
        kw.strip() for kw in capability.split(",")
        if kw.strip()
    ]
    seen = set()
    keywords = [kw for kw in keywords if not (kw in seen or seen.add(kw))]

    if not keywords:
        return {"success": False, "error": "capability 不能为空"}

    tried = _get_tried_servers()
    logger.info(
        "[dsh][discovery] search_and_connect_mcp: keywords=%r tried_servers=%r",
        keywords, tried,
    )

    candidates, meta = await search_configs_multi(keywords, limit=limit)

    if not candidates:
        if meta.get("network_error"):
            return {
                "success": False,
                "error": "MCP Registry 网络不可达（重试后仍失败）。",
                "network_error": True,
                "tried_keywords": meta.get("tried_keywords", []),
            }
        return {
            "success": False,
            "error": (
                f"未在 MCP Registry 中找到与 {keywords!r} 相关的服务器。"
                f"尝试过的关键词：{', '.join(meta.get('tried_keywords', []))}"
            ),
            "tried_keywords": meta.get("tried_keywords", []),
        }

    tried = _get_tried_servers()
    fresh_candidates = [
        c for c in candidates
        if c["registry_name"] not in tried
    ]

    if not fresh_candidates:
        logger.info(
            "[dsh][discovery] all %d candidates already tried, resetting",
            len(candidates),
        )
        reset_tried_servers()
        tried = _get_tried_servers()
        fresh_candidates = list(candidates)

    logger.info(
        "[dsh][discovery] %d fresh candidates (excluded %d tried)",
        len(fresh_candidates), len(candidates) - len(fresh_candidates),
    )

    chosen = await _rank_candidates(capability, fresh_candidates)
    if chosen is None:
        return {
            "success": False,
            "error": (
                f"搜索到 {len(fresh_candidates)} 个候选，但都不相关。"
                f"候选前 5 个：{[c['registry_name'] for c in fresh_candidates[:5]]}。"
                f"**下一步：换一个更宽的关键词再搜索**。"
            ),
            "candidates": [c["registry_name"] for c in fresh_candidates[:10]],
        }

    manager = _get_manager()
    local_name = chosen["local_name"]

    if manager.has_server(local_name) and local_name in manager.connections:
        tools = [
            spec.public_name
            for spec in manager.tools.values()
            if spec.server_name == local_name
        ]
        return {
            "success": True,
            "server": chosen["registry_name"],
            "transport": chosen["transport"],
            "url": chosen["url"],
            "tools": tools,
            "message": _format_tools_for_message(tools),
            "matched_keyword": meta.get("matched_keyword"),
            "note": (
                "这是之前已连接的服务器。如果它的工具无法满足需求，"
                "请再次调用 search_and_connect_mcp 用更宽的关键词，"
                "系统会选一个不同的候选。"
            ),
        }

    config = await config_for_entry(chosen)
    if config is None:
        return {
            "success": False,
            "error": f"无法为 {chosen['registry_name']} 构造有效的服务器配置",
        }

    manager.register_config(config)
    try:
        await manager.connect(local_name)
        await manager.discover(local_name, _register_tool, _unregister_tool)
    except Exception as error:
        _get_tried_servers().add(chosen["registry_name"])
        manager.unregister_config(local_name)
        return {
            "success": False,
            "error": f"连接 {chosen['registry_name']} 失败: {error}",
        }

    _get_tried_servers().add(chosen["registry_name"])

    tools = [
        spec.public_name
        for spec in manager.tools.values()
        if spec.server_name == local_name
    ]
    return {
        "success": True,
        "server": chosen["registry_name"],
        "transport": chosen["transport"],
        "url": chosen["url"],
        "tools": tools,
        "message": _format_tools_for_message(tools),
        "matched_keyword": meta.get("matched_keyword"),
    }