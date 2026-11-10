"""Test: essential_when tools sort first after prerequisite completion."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools.orchestration.core import AgentOrchestrator


def _make_orch():
    agent = MagicMock()
    agent.list_all_tools = lambda **kw: {}
    llm = MagicMock()
    llm.generate = lambda *a, **kw: '{"thought":"done","done":true}'
    sl = MagicMock()
    sl.shortlist = lambda goal, tools, **kw: dict(tools)
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


def _make_tool(name, essential_when=()):
    m = MagicMock()
    m.description = name
    m.essential_when = essential_when
    m.input_schema = {"type": "object", "properties": {}, "required": []}
    m.parameters = {}
    m.required = []
    m.danger_level = "safe"
    m.trust_level = 0
    m.origin = "builtin"
    return m


def test_essential_when_sorted_first():
    """有 essential_when 的工具排在无 essential_when 的前面。"""
    orch = _make_orch()
    tools = {
        "write_docx": _make_tool("write_docx", essential_when=("generate", "document")),
        "convert_to_pdf": _make_tool("convert_to_pdf", essential_when=("convert", "pdf")),
        "list_files": _make_tool("list_files", essential_when=()),
        "read_text_file": _make_tool("read_text_file", essential_when=()),
    }
    picked = orch._select_shortlist("\u751f\u6210 PDF", tools)
    keys = list(picked.keys())
    # 有 essential_when 的在前
    essential = [k for k in keys if getattr(picked[k], "essential_when", ())]
    non_essential = [k for k in keys if not getattr(picked[k], "essential_when", ())]
    if essential and non_essential:
        assert keys.index(essential[0]) < keys.index(non_essential[0]), (
            f"essential tools should come first, got {keys}"
        )


def test_essential_when_stable_order():
    """同组内按名字字典序，保证可复现。"""
    orch = _make_orch()
    tools = {
        "z_tool": _make_tool("z_tool", essential_when=("generate",)),
        "a_tool": _make_tool("a_tool", essential_when=("generate",)),
        "m_tool": _make_tool("m_tool"),
    }
    picked = orch._select_shortlist("\u751f\u6210", tools)
    keys = list(picked.keys())
    # 有 essential 的 a/z 在前，a < z
    assert "m_tool" in keys
    if "a_tool" in keys and "z_tool" in keys:
        assert keys.index("a_tool") < keys.index("z_tool")
