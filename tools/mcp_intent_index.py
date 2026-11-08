"""MCP server/tool 的「语义-工具」离线索引（纯本地 embedding，零 LLM）。

设计（重构后）：
    - 编码文本 = tool_name + description + capability_tags，不再用 LLM 合成意图
    - build_index_full / build_index_incremental：首次全量、加 server 增量
    - load_or_build_index：工具集不变时直接加载 records.json（含 vector），零编码
    - search_intents：goal 编码 → FAISS top-k → 相似度阈值过滤 → 按 server 聚合

和旧版的区别：
    - 无 LLM 合成意图（编码 name + description + capability_tags）
    - records.json 持久化 vector 字段，启动零编码
    - 加相似度阈值（默认 0.5），避免 fetch 之类弱相关工具被召回

不适用范围：
    - CapabilityIndex（tools/capability/index.py）是另一套索引，本模块不碰
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from infra.embedding import get_encoder

logger = logging.getLogger(__name__)

_INDEX_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "mcp_intent_index",
)
_INDEX_FILE = os.path.join(_INDEX_DIR, "faiss.index")
_RECORDS_FILE = os.path.join(_INDEX_DIR, "records.json")

_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# 相似度阈值：低于此值的候选丢弃。
# bge-small-zh 经验值：0.7+ 强相关，0.5~0.7 相关，0.3~0.5 弱相关。
# 0.5 能召回同义表达，又不会把 fetch 之类无关工具带进来。
_DEFAULT_MIN_SCORE = 0.5


# ─────────────────────────────────────────────────────────────
# 文本构造
# ─────────────────────────────────────────────────────────────

def _build_tool_text(spec: Any) -> str:
    """把一个 MCPToolSpec 拼成"语义指纹"文本。

    格式：`{remote_name} {description} {capability_tags 空格拼接}`
    例：`create_pptx Create a PowerPoint presentation... generate presentation ppt pptx slide`
    """
    name = getattr(spec, "remote_name", "") or getattr(spec, "public_name", "")
    desc = getattr(spec, "description", "") or ""
    tags = getattr(spec, "capability_tags", ()) or ()
    tag_str = " ".join(str(t) for t in tags)
    return f"{name} {desc} {tag_str}".strip()


def _iter_server_tools(manager):
    """按 server 分组遍历 manager.tools（MCPToolSpec 字典）。"""
    by_server: Dict[str, List] = {}
    for spec in manager.tools.values():
        by_server.setdefault(spec.server_name, []).append(spec)
    for name, specs in by_server.items():
        yield name, specs


def _make_record(spec: Any, vector: List[float]) -> Dict[str, Any]:
    return {
        "text": _build_tool_text(spec),
        "vector": vector,
        "server_name": spec.server_name,
        "tool_name": spec.public_name,
        "remote_name": spec.remote_name,
    }


# ─────────────────────────────────────────────────────────────
# 编码 + 写盘
# ─────────────────────────────────────────────────────────────

def _encode_texts(texts: List[str]) -> List[List[float]]:
    """批量编码。bge-small 对 94 条短文本约 100~300ms。"""
    encoder = get_encoder(_EMBED_MODEL, local_files_only=True)
    vecs = encoder.encode(
        texts, normalize_embeddings=True, show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


def _write_index_atomic(records: List[Dict[str, Any]]) -> None:
    """原子写 records.json + faiss.index。records 里必须含 vector。"""
    if not records:
        logger.info("[mcp_intent] no records, skip write")
        return

    os.makedirs(_INDEX_DIR, exist_ok=True)

    import faiss
    import numpy as np

    vectors = [r["vector"] for r in records]
    arr = np.array(vectors, dtype="float32")
    dim = arr.shape[1]

    index = faiss.IndexFlatIP(dim)   # 归一化后 IP = cosine
    index.add(arr)

    fd, tmp_records = tempfile.mkstemp(dir=_INDEX_DIR, suffix=".records.json.tmp")
    os.close(fd)
    with open(tmp_records, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    fd, tmp_index = tempfile.mkstemp(dir=_INDEX_DIR, suffix=".faiss.index.tmp")
    os.close(fd)
    faiss.write_index(index, tmp_index)

    os.replace(tmp_records, _RECORDS_FILE)
    os.replace(tmp_index, _INDEX_FILE)
    logger.info(
        "[mcp_intent] index written: %d records, dim=%d", len(records), dim,
    )


# ─────────────────────────────────────────────────────────────
# 全量 / 增量 / 加载
# ─────────────────────────────────────────────────────────────

def build_index_full(manager) -> Dict[str, Any]:
    """全量重建：编码 manager.tools 里所有工具。"""
    specs = list(manager.tools.values())
    if not specs:
        logger.info("[mcp_intent] no tools registered, skip")
        return {"indexed_tools": 0, "indexed_records": 0}

    texts = [_build_tool_text(s) for s in specs]
    vectors = _encode_texts(texts)
    records = [_make_record(s, v) for s, v in zip(specs, vectors)]

    _write_index_atomic(records)
    logger.info(
        "[mcp_intent] full build: %d tools, %d records",
        len(specs), len(records),
    )
    return {"indexed_tools": len(specs), "indexed_records": len(records)}


def build_index_incremental(manager) -> Dict[str, Any]:
    """增量重建：只编码新增工具，剔除已删除工具，复用已有 vector。"""
    old_records = _load_records()
    old_by_tool = {r["tool_name"]: r for r in old_records}

    current_specs = list(manager.tools.values())
    current_names = {s.public_name for s in current_specs}

    # 只编码"新增的工具"
    new_specs = [s for s in current_specs if s.public_name not in old_by_tool]
    new_records: List[Dict[str, Any]] = []
    if new_specs:
        texts = [_build_tool_text(s) for s in new_specs]
        vectors = _encode_texts(texts)
        new_records = [_make_record(s, v) for s, v in zip(new_specs, vectors)]
        logger.info(
            "[mcp_intent] incremental: %d new tools encoded", len(new_specs),
        )

    # 保留仍然存在的旧记录（vector 复用，零编码）
    kept = [r for r in old_records if r["tool_name"] in current_names]

    merged = kept + new_records
    if not merged:
        logger.info("[mcp_intent] incremental: empty after merge, skip")
        return {"indexed_tools": 0, "indexed_records": 0}

    _write_index_atomic(merged)
    logger.info(
        "[mcp_intent] incremental done: kept %d, added %d, total %d",
        len(kept), len(new_records), len(merged),
    )
    return {"indexed_tools": len(current_specs), "indexed_records": len(merged)}


def _load_records() -> List[Dict[str, Any]]:
    if not os.path.exists(_RECORDS_FILE):
        return []
    try:
        with open(_RECORDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        logger.exception("[mcp_intent] load records failed")
        return []


def load_or_build_index(manager) -> Dict[str, Any]:
    """工具集不变 → 直接加载（零编码）；变了 → 增量；无索引 → 全量。

    这是启动时调用的入口，应该放在 asyncio.to_thread 里跑。
    """
    current_names = {s.public_name for s in manager.tools.values()}
    if not current_names:
        logger.info("[mcp_intent] no tools, skip index")
        return {"indexed_tools": 0, "indexed_records": 0}

    old_records = _load_records()
    old_names = {r.get("tool_name") for r in old_records}

    has_vector = old_records and all("vector" in r for r in old_records)
    has_index_file = os.path.exists(_INDEX_FILE)

    if old_names == current_names and has_vector and has_index_file:
        logger.info(
            "[mcp_intent] index hit: %d tools, no rebuild needed", len(old_names),
        )
        return {"indexed_tools": len(old_names), "indexed_records": len(old_records)}

    if old_records and has_vector:
        logger.info(
            "[mcp_intent] tool set changed (%d → %d), incremental rebuild",
            len(old_names), len(current_names),
        )
        return build_index_incremental(manager)

    logger.info("[mcp_intent] no usable index, full build")
    return build_index_full(manager)


# ─────────────────────────────────────────────────────────────
# 检索
# ─────────────────────────────────────────────────────────────

def search_intents(
    query: str,
    top_k: int = 20,
    per_server_cap: int = 3,
    min_score: float = _DEFAULT_MIN_SCORE,
    frame=None,
) -> List[Dict[str, Any]]:
    """语义检索工具索引，返回按 server 聚合的候选。

    B6-c 起委托给 tools.semantics.index.strategies.intent_search。
    保留原签名，调用方零改动。
    """
    from tools.semantics.index.strategies.intent_search import (
        search_intents as _impl,
    )
    return _impl(
        query,
        _INDEX_DIR,
        top_k=top_k,
        per_server_cap=per_server_cap,
        min_score=min_score,
        frame=frame,
    )


def clear_index() -> None:
    """清空索引文件（测试/调试用）。"""
    for path in (_RECORDS_FILE, _INDEX_FILE):
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                logger.exception("[mcp_intent] remove %s failed", path)
