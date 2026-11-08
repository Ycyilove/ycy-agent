"""Discovery tools that let the orchestrator search the public MCP Registry."""

from __future__ import annotations

import asyncio
import logging

import os
import re
from contextvars import ContextVar
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set

from . import tool

# ── per-task tried servers（async-safe） ──
# 原来挂在 manager.tried_servers 上，是跨任务共享的；
# 改成 contextvars，避免任务 A 的 discovery 被任务 B 的 reset 清空。
_tried_servers_var: ContextVar[FrozenSet[str]] = ContextVar(
    "mcp_tried_servers", default=frozenset()
)


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


def _get_tried_servers() -> FrozenSet[str]:
    """返回当前任务的 tried servers（frozenset，不可变）。"""
    return _tried_servers_var.get()


def _mark_tried(name: str) -> None:
    """把 name 加入当前任务的 tried 集合。"""
    current = _tried_servers_var.get()
    _tried_servers_var.set(current | {name})


def reset_tried_servers() -> None:
    """重置当前任务的 tried servers。

    用 contextvars，per-task 隔离，不影响并发任务。
    """
    _tried_servers_var.set(frozenset())


# ─────────────────────────────────────────────────────────────
# 启发式排序（不调 LLM）
# ─────────────────────────────────────────────────────────────

# 地理相关的关键词（只做地理匹配，不含任何业务领域词）
_GEO_US_ONLY = (
    "united states", "us-only", "us only",
)
_GEO_CHINA = (
    "china", "chinese",
)
_GEO_GLOBAL = (
    "global", "worldwide", "any location",
)

# ── 关键词扩展（纯通用规则，不含任何领域词表） ──

# 常见通用后缀：剥离后得到更宽的主体词
# 这些后缀是 MCP 包命名里的**结构约定**，与领域无关
_STRIP_SUFFIXES = (
    "-data", "-api", "-server", "-service", "-mcp", "-tool",
    "-provider", "-source", "-feed", "-hub", "-client", "-adapter",
    "_data", "_api", "_server", "_service", "_mcp", "_tool",
    "_provider", "_source", "_feed", "_hub", "_client",
)


def _expand_keywords(keywords: List[str]) -> List[str]:
    """扩展关键词列表。

    纯规则，不依赖任何领域词表：
        1. 保留原始词
        2. 剥离常见**结构后缀**（-data / -api / -server / ...）
        3. 多段词（>=3 段）截短为前 2 段

    例：
        ["foo-bar-data"]     → ["foo-bar-data", "foo-bar"]
        ["foo_bar_service"]  → ["foo_bar_service", "foo_bar"]
    """
    expanded: List[str] = []
    seen: set = set()

    def _add(kw: str) -> None:
        kw = kw.strip().lower()
        # 太短（<3）或空的丢掉，避免噪声
        if kw and len(kw) >= 3 and kw not in seen:
            seen.add(kw)
            expanded.append(kw)

    # 1. 原始词
    for kw in keywords:
        _add(kw)

    # 2. 剥离通用后缀（每个词只剥一次）
    for kw in list(keywords):
        lower = kw.lower()
        for suffix in _STRIP_SUFFIXES:
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                _add(lower[: -len(suffix)])
                break

    # 3. 多段词截短（>=3 段 → 前 2 段）
    for kw in list(keywords):
        lower = kw.lower()
        for sep in ("-", "_", " "):
            if sep in lower:
                parts = [p for p in lower.split(sep) if p]
                if len(parts) >= 3:
                    _add(sep.join(parts[:2]))
                break

    return expanded[:8]

# ── 自动关键词生成（capability 为空时的兜底） ──

_AUTO_KW_PROMPT = """把用户问题转换成 2~4 个英文搜索关键词，用于 MCP Registry 搜索。

规则：
- 只输出英文关键词，逗号分隔
- 用 MCP 作者会在包名里用的通用词根，不要逐字翻译
- 从最具体到最宽泛排序
- 只输出关键词，不要解释

用户问题：{query}

关键词："""


def _auto_generate_keywords(user_query: str) -> List[str]:
    """用 LLM 把 user_query 转成英文关键词。失败返回空列表。"""
    if _llm_provider is None:
        return []
    try:
        raw = _llm_provider().generate(
            [{"role": "user", "content": _AUTO_KW_PROMPT.format(query=user_query)}],
            mode="quick",
        )
        raw = (raw or "").strip().strip("`").strip()
        if "\n" in raw:
            raw = raw.split("\n")[0].strip()
        tokens = [
            t.strip().lower()
            for t in raw.replace("，", ",").split(",")
            if t.strip()
        ]
        # 只保留 ASCII token（Registry 只接受英文）
        tokens = [
            t for t in tokens
            if all(c.isascii() for c in t) and len(t) >= 2
        ]
        return tokens[:4]
    except Exception:
        logger.exception("[dsh][discovery] auto keyword generation failed")
        return []

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
        1. 启发式排序，用于稳定排序 & 截断
        2. 只要候选数合理（<= _LLM_RANK_TOP_N），都交给 LLM 做领域判断
        3. 候选数过多时先截断到 top N，再交给 LLM
        4. LLM 不可用时退到启发式 Top 1
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
    # 关键词匹配天然有歧义（同一个词在不同领域含义不同）。
    # 启发式无法判断语义相关性，只要候选数合理，都交给 LLM 做领域判断。
    if _llm_provider is None:
        logger.info(
            "[dsh][discovery] no LLM provider, using heuristic top 1: %s",
            ranked[0]["registry_name"],
        )
        return ranked[0]

    # 候选数过多时，先启发式取 top N，缩短 prompt
    if len(ranked) > _LLM_RANK_TOP_N:
        logger.info(
            "[dsh][discovery] %d candidates > %d, pre-truncating",
            len(ranked), _LLM_RANK_TOP_N,
        )
        ranked = ranked[:_LLM_RANK_TOP_N]
    else:
        logger.info(
            "[dsh][discovery] %d candidates, calling LLM for domain matching",
            len(ranked),
        )

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
        "**优先级 1：领域匹配（最重要）。**\n"
        "   - capability 里的词可能是**多义词**——同一个词在不同领域\n"
        "     有完全不同的含义。\n"
        "   - 必须读每个候选的 description，判断它描述的业务领域\n"
        "     是否和用户的实际需求一致。\n"
        "   - **名字里含有关键词，但 description 里的领域完全不同 → 排除。**\n"
        "   - 宁可返回 0，也不要选一个领域不符的候选。\n\n"
        "**优先级 2：地理覆盖匹配。**\n"
        "   - 用户需求涉及具体地区时，看 description 里的地理范围是否匹配。\n\n"
        "**优先级 3：功能匹配。**\n\n"
        "**优先级 4：无需认证优先。**\n\n"
        "只返回最相关的那一个服务器的序号（纯数字）。\n"
        "如果所有候选**领域不符或都需要认证**，返回 0"
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
        # LLM 说"都不相关"，返回 None，让上层决定要不要继续尝试其它候选
        return None
    return top_n[idx - 1]


def _format_tools_for_message(tools: List[str]) -> str:
    if not tools:
        return "（该服务器没有暴露任何工具）"
    lines = [
        "**发现以下可用工具（候选，请根据用户目标判断是否使用）**："
    ]
    for t in tools:
        lines.append(f"  - {t}")
    lines.append("")
    lines.append(
        "**调用时请使用上面的完整工具名**，格式为 mcp__<server>__<tool>。"
        "**不要自己构造或简化工具名**。"
        "如果这些工具与当前目标明显不符，可以忽略它们并选择其他更合适的工具。"
    )
    return "\n".join(lines)


@tool(
    name="search_and_connect_mcp",
    description=(
        "Search the public MCP Registry for a server that provides the given "
        "capability, connect to it, and return the tools it exposes. "
        "Use this when the user needs external data or actions you cannot do "
        "with existing tools.\n\n"
        "**CRITICAL — `user_query` is REQUIRED and must be the user's "
        "ORIGINAL wording in their own language (Chinese is fine), "
        "verbatim, not translated, not summarized.** "
        "The local intent index matches against `user_query` first; "
        "if it is missing or paraphrased, discovery degrades badly.\n\n"
        "**`capability` is a FALLBACK**: only used when the local intent "
        "index has no hit. Must be English keywords, comma-separated, "
        "ordered from most specific to most general.\n\n"
        "**IMPORTANT: Each call picks a DIFFERENT candidate.** "
        "Servers already connected in this session are excluded from future "
        "searches. If the currently connected server does NOT match the "
        "user's needs, **call this tool again with a broader capability**.\n\n"
        "**After connecting**: read the returned `description` field and "
        "verify the server is in the **same domain** as the user's request.\n\n"
        "**After this tool returns, use the exact tool names from the "
        "`tools` list. Do NOT construct or guess tool names.**"
    ),
    parameters={
        "user_query": {
            "type": "str",
            "description": (
                "**必填**。用户的**原始问题原文**，保持用户的语言（中文可以），"
                "逐字照抄，不要翻译、不要改写、不要总结。"
            ),
        },
        "capability": {
            "type": "str",
            "description": (
                "**可选**。英文能力关键词，逗号分隔多个候选。"
                "只在 user_query 本地索引无命中时作为回退使用。"
                "**禁止传中文**。"
            ),
        },
        "limit": {
            "type": "int",
            "description": "每个关键词的搜索候选条数上限，默认 100",
        },
    },
    required=["user_query"],
    examples=[
        "search_and_connect_mcp(user_query='实时查询维尼斯当前状态以及外界评价', capability='news,realtime')",
        "search_and_connect_mcp(user_query='帮我找一下最新的 AI 新闻', capability='news,search')",
    ],
    category="mcp",
    danger_level="safe",
)
async def search_and_connect_mcp(
    user_query: str = "",
    capability: str = "",
    limit: int = 100,
) -> Dict[str, Any]:
    """搜索 Registry + 连接候选（B7-c 薄壳）。

    委托给 tools.capability.pipeline.search_and_connect_mcp（registry 模式 7 阶段）。
    """
    from tools.capability.pipeline import (
        search_and_connect_mcp as _impl,
    )
    return await _impl(
        user_query=user_query,
        capability=capability,
        limit=limit,
    )


# ── 异步索引刷新（不阻塞 discovery 返回） ──

_intent_index_refresh_lock: Optional[asyncio.Lock] = None


def _schedule_intent_index_refresh() -> None:
    """连接成功后，后台为新 server 增量重建索引。

    重构后：底层走 mcp_intent_index.load_or_build_index（本地 embedding +
    增量更新），不再调 LLM 合成意图。工具集变了自动增量编码新增工具。

    不阻塞当前协程；用 to_thread 把编码 + FAISS 重建放到线程池。
    失败只记日志，不影响 discovery 结果。
    """
    global _intent_index_refresh_lock
    if _intent_index_refresh_lock is None:
        _intent_index_refresh_lock = asyncio.Lock()

    async def _do_refresh():
        async with _intent_index_refresh_lock:
            try:
                from .mcp_intent_index import load_or_build_index
                manager = _get_manager()
                await asyncio.to_thread(load_or_build_index, manager)
                logger.info("[dsh][discovery] intent index refreshed")
            except Exception:
                logger.exception(
                    "[dsh][discovery] intent index refresh failed",
                )

    try:
        asyncio.create_task(_do_refresh())
    except RuntimeError:
        # 没有 running loop（同步上下文）→ 跳过
        logger.debug("[dsh][discovery] no running loop, skip index refresh")

# 公开别名（供 backend/routes/mcp.py 等外部调用）
schedule_intent_index_refresh = _schedule_intent_index_refresh

# ── SearchPort 适配器（供 core.ports 注入） ──

class _SearchAdapter:
    """把 mcp_discovery 的 tavily/fetch 兜底搜索适配成 core.ports.SearchPort。"""

    async def search_web(self, query: str, max_results: int = 5):
        """尝试 tavily -> fetch。返回已格式化文本；失败返回 None。"""
        try:
            manager = _get_manager()
        except Exception:
            logger.exception("[dsh][search_port] manager 不可用")
            return None

        for srv in ("tavily", "fetch"):
            if srv not in getattr(manager, "configs", {}):
                continue
            try:
                if srv not in manager.connections:
                    await manager.connect(srv)
                    await manager.discover(srv, _register_tool, _unregister_tool)
                if srv == "tavily":
                    payload = await manager.call_tool(
                        "mcp__tavily__tavily_search",
                        {"query": query, "max_results": max_results},
                    )
                else:
                    payload = await manager.call_tool(
                        "mcp__fetch__fetch", {"url": query},
                    )
                import json as _json
                return _json.dumps(
                    payload, ensure_ascii=False, default=str
                )[:6000]
            except Exception:
                logger.exception(
                    "[dsh][search_port] search via %s failed", srv
                )
        return None


def make_search_port():
    """bootstrap 注入用。"""
    return _SearchAdapter()
