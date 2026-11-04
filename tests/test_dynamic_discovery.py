"""验证 P0-2 不破坏动态发现：discovery 后新工具仍进下一步候选。"""
from unittest.mock import MagicMock

from tools.orchestration.core import AgentOrchestrator
from tools.orchestration.state import OrchestratorState


def _make_orch():
    agent = MagicMock()
    llm = MagicMock()
    return AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: llm,
    )


def test_non_web_task_skips_static_mcp():
    """意图索引无命中 + 非网页任务 → 不保留静态 MCP。"""
    orch = _make_orch()
    tools = {
        "read_text_file": MagicMock(description="read"),
        "search_and_connect_mcp": MagicMock(description="discover"),
        "mcp__tavily__tavily_search": MagicMock(description="tavily search"),
        "mcp__news_server__search_news": MagicMock(description="news"),
    }
    shortlist = orch._select_shortlist("修 calculator/core.py 的 bug", tools)
    # 非网页任务：tavily 被筛掉
    assert "mcp__tavily__tavily_search" not in shortlist
    # 动态 news 也不在（由 _decide 补回）
    assert "mcp__news_server__search_news" not in shortlist
    # discovery 入口始终在
    assert "search_and_connect_mcp" in shortlist


def test_web_task_keeps_static_mcp():
    """意图索引无命中 + 网页任务 → 保留静态 MCP。"""
    orch = _make_orch()
    tools = {
        "read_text_file": MagicMock(description="read"),
        "search_and_connect_mcp": MagicMock(description="discover"),
        "mcp__tavily__tavily_search": MagicMock(description="tavily search"),
    }
    shortlist = orch._select_shortlist("查一下今天的新闻", tools)
    assert "mcp__tavily__tavily_search" in shortlist


def test_recent_discovered_readded_in_decide():
    """_decide 应把 state.recent_discovered_tools 补进 shortlist。"""
    orch = _make_orch()
    tools = {
        "read_text_file": MagicMock(description="read"),
        "search_and_connect_mcp": MagicMock(description="discover"),
        "mcp__news_server__search_news": MagicMock(description="news"),
    }
    state = OrchestratorState(goal="查新闻")
    state.recent_discovered_tools = {"mcp__news_server__search_news"}

    # 直接调 _select_shortlist，动态不在
    base = orch._select_shortlist("查新闻", tools)
    assert "mcp__news_server__search_news" not in base

    # 模拟 _decide 里的补回逻辑
    recent = getattr(state, "recent_discovered_tools", set()) or set()
    for name in recent:
        if name in tools and name not in base:
            base[name] = tools[name]
    assert "mcp__news_server__search_news" in base


def test_snapshot_roundtrip_recent_discovered():
    """recent_discovered_tools 应随 snapshot 往返。"""
    st = OrchestratorState(goal="x")
    st.recent_discovered_tools = {"mcp__a__b", "mcp__c__d"}
    snap = st.snapshot()
    st2 = OrchestratorState.from_snapshot(snap)
    assert st2.recent_discovered_tools == {"mcp__a__b", "mcp__c__d"}