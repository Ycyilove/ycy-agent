"""Test: secondary_ordered sorted by format-word first position.

Source is pure ASCII (Chinese via \\u escapes) to avoid PowerShell encoding.
"""
from __future__ import annotations

import pytest

from tools.semantics.intent import (
    IntentSet,
    _classify_single,
    _match_formats_with_position,
    classify,
)


def test_match_formats_with_position_basic():
    # "\u751f\u6210\u4e00\u4e2a PPT\uff0c\u53e6\u5916\u63d0 Excel"
    goal = "\u751f\u6210\u4e00\u4e2a PPT\uff0c\u53e6\u5916\u63d0 Excel"
    hits = _match_formats_with_position(goal)
    assert hits, "should match at least one format"
    fmts = [f for f, _pos in hits]
    assert "presentation" in fmts
    assert "spreadsheet" in fmts
    assert fmts.index("presentation") < fmts.index("spreadsheet")


def test_match_formats_with_position_empty():
    assert _match_formats_with_position("") == []
    assert _match_formats_with_position("hello world") == []


def test_match_formats_with_position_sorted():
    # "Excel \u548c PPT \u90fd\u60f3\u8981"
    goal = "Excel \u548c PPT \u90fd\u60f3\u8981"
    hits = _match_formats_with_position(goal)
    fmts = [f for f, _pos in hits]
    assert "spreadsheet" in fmts and "presentation" in fmts
    assert fmts.index("spreadsheet") < fmts.index("presentation")


def test_classify_single_secondary_ordered():
    # "\u751f\u6210\u4e00\u4e2a PPT\uff0c\u53e6\u5916\u63d0 Excel\u3001Word"
    goal = "\u751f\u6210\u4e00\u4e2a PPT\uff0c\u53e6\u5916\u63d0 Excel\u3001Word"
    it = _classify_single(goal)
    assert isinstance(it, IntentSet)
    assert it.primary == "generate", f"primary={it.primary!r}"
    assert it.secondary_ordered, "secondary_ordered should be non-empty"
    ordered_fmts = [f for f, _pos in it.secondary_ordered]
    assert ordered_fmts[0] == "presentation", f"got {ordered_fmts}"


def test_classify_single_secondary_still_frozenset():
    # "\u751f\u6210 PPT \u548c Excel"
    goal = "\u751f\u6210 PPT \u548c Excel"
    it = _classify_single(goal)
    assert isinstance(it.secondary, frozenset)
    assert "presentation" in it.secondary
    assert "spreadsheet" in it.secondary


def test_classify_segments_aggregates_ordered():
    # long goal: "\u751f\u6210\u4e00\u4e2a PPT\uff0c\u5173\u4e8e AI \u53d1\u5c55\u3002" + "x"*500 + "\u53e6\u5916\u8981 Excel \u8868\u683c\u3002"
    long_goal = (
        "\u751f\u6210\u4e00\u4e2a PPT\uff0c\u5173\u4e8e AI \u53d1\u5c55\u3002"
        + "x" * 500
        + "\u53e6\u5916\u8981 Excel \u8868\u683c\u3002"
    )
    it = classify(long_goal)
    assert it.secondary_ordered, "aggregated secondary_ordered should be non-empty"
    ordered_fmts = [f for f, _pos in it.secondary_ordered]
    assert ordered_fmts[0] == "presentation", f"got {ordered_fmts}"


def test_intentset_backward_compat():
    it = IntentSet(
        tags=frozenset({"generate"}),
        primary="generate",
        secondary=frozenset({"presentation"}),
    )
    assert it.secondary_ordered == ()
    assert "presentation" in it.secondary
