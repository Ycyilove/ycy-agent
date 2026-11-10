"""Test: ShortlistStrategy top_k / weight_overrides params."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools.semantics.index.strategies.shortlist import ShortlistStrategy


def _make_tools(n: int):
    out = {}
    for i in range(n):
        m = MagicMock()
        m.description = f"tool {i}"
        out[f"tool_{i}"] = m
    return out


def test_shortlist_accepts_top_k():
    strat = ShortlistStrategy(top_k=3)
    assert hasattr(strat, "shortlist")
    import inspect
    sig = inspect.signature(strat.shortlist)
    assert "top_k" in sig.parameters, f"shortlist should accept top_k, got {list(sig.parameters)}"
    assert "weight_overrides" in sig.parameters, "shortlist should accept weight_overrides"


def test_shortlist_module_level_passthrough():
    import importlib
    import inspect
    # strategies/__init__.py 把 shortlist 函数导出了，from ...strategies import shortlist
    # 拿到的是函数（不是模块），要 importlib 取模块
    mod = importlib.import_module("tools.semantics.index.strategies.shortlist")
    sig = inspect.signature(mod.shortlist)
    assert any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ), f"module-level shortlist should accept **kwargs, got {list(sig.parameters)}"
