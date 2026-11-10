"""Test: scene soft-filter (downweight not drop)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools.orchestration.core import AgentOrchestrator


def _make_orch(shortlist_returns=None):
    agent = MagicMock()
    agent.list_all_tools = lambda **kw: {}
    llm = MagicMock()
    llm.generate = lambda *a, **kw: '{"thought":"done","done":true}'

    # mock shortlister：返回全部工具（模拟 embedding 失败/全量）
    sl = MagicMock()
    def _sl(goal, tools, **kwargs):
        if shortlist_returns is not None:
            return shortlist_returns
        return dict(tools)
    sl.shortlist = _sl

    try:
        return AgentOrchestrator(
            agent_provider=lambda: agent,
            llm_provider=lambda: llm,
            shortlister_provider=lambda: sl,
        )
    except TypeError:
        return AgentOrchestrator(
            agent_provider=lambda: agent,
            llm_provider=lambda: llm,
        )


def _make_tool(name):
    m = MagicMock()
    m.description = name
    m.essential_when = ()
    m.input_schema = {"type": "object", "properties": {}, "required": []}
    m.parameters = {}
    m.required = []
    m.danger_level = "safe"
    m.trust_level = 0
    m.origin = "builtin"
    return m


def test_scene_blocklist_exists():
    """_scene_blocklist 是 classmethod/staticmethod。"""
    orch = _make_orch()
    assert hasattr(orch, "_scene_blocklist") or hasattr(
        type(orch), "_scene_blocklist"
    ), "AgentOrchestrator should have _scene_blocklist"


def test_scene_blocklist_data_scene():
    """'生成 PPT，另外提 Excel' -> scene=data -> blocklist 含 write_excel。"""
    orch = _make_orch()
    bl = orch._scene_blocklist("\u751f\u6210 PPT\uff0c\u53e6\u5916\u63d0 Excel")
    assert isinstance(bl, set), f"_scene_blocklist should return set, got {type(bl)}"
    assert "write_excel" in bl or "edit_file" in bl, (
        f"data scene blocklist should contain write_excel/edit_file, got {bl}"
    )


def test_scene_blocklist_code_scene():
    orch = _make_orch()
    bl = orch._scene_blocklist("\u4fee\u590d bug\uff0c\u8fd0\u884c pytest")
    assert "mysql_query" in bl or "read_csv" in bl, (
        f"code scene blocklist should contain mysql_query/read_csv, got {bl}"
    )


def test_scene_blocklist_no_scene():
    orch = _make_orch()
    bl = orch._scene_blocklist("hello world")
    assert bl == set(), f"no scene should return empty set, got {bl}"


def test_select_shortlist_soft_filter_keeps_write_excel():
    """'生成 PPT，另外提 Excel' -> write_excel 不应被硬剔除（软过滤）。"""
    orch = _make_orch()
    tools = {
        "create_pptx": _make_tool("create_pptx"),
        "write_excel": _make_tool("write_excel"),
        "write_file": _make_tool("write_file"),
        "read_csv": _make_tool("read_csv"),
    }
    picked = orch._select_shortlist(
        "\u751f\u6210 PPT\uff0c\u53e6\u5916\u63d0 Excel", tools
    )
    # 软过滤：write_excel 不应被硬剔除
    assert "write_excel" in picked, (
        f"write_excel should be kept (soft filter), got {list(picked.keys())}"
    )
