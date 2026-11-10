"""对外统一 API。三策略 + 编码/存储入口。

用法：
    from tools.semantics.index.facade import (
        shortlist_tools,
        search_intents,
        hybrid_search,
        build_tool_text,
        encode_texts,
        encode_query,
        write_index_atomic,
        load_records,
        load_index,
    )
"""
from __future__ import annotations

from .encoder import build_tool_text, encode_query, encode_texts
from .store import (
    fingerprint,
    load_index,
    load_records,
    write_index_atomic,
)
from .strategies.shortlist import shortlist as _shortlist
from .strategies.intent_search import search_intents
from .strategies.hybrid import hybrid_search


def shortlist_tools(goal, tools):
    """goal → top-k 工具（失败返回 None）。"""
    return _shortlist(goal, tools)


__all__ = [
    "build_tool_text",
    "encode_query",
    "encode_texts",
    "fingerprint",
    "write_index_atomic",
    "load_records",
    "load_index",
    "shortlist_tools",
    "search_intents",
    "hybrid_search",
]