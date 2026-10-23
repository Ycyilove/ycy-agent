"""MCP server/tool 的「意图-工具」离线索引。

对每个已注册 MCP server 的每个 tool，用 LLM 合成 N 条多样化口语意图，
连同 tool description 一起写入 FAISS + BM25，供在线检索使用。
"""

from __future__ import annotations

import json
import logging
import os
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
_SYNTH_INTENTS_PER_TOOL = 8

_SYNTH_PROMPT = """你是用户意图生成助手。给定一个 MCP 工具的描述，生成 {n} 条**中文口语化**的用户意图。

工具名：{tool_name}
工具描述：{description}

要求：
1. 每条意图是一句用户可能说的话，口语化，像真人提问
2. 覆盖不同表达方式（"我想…""帮我…""能不能…""实时看看…"）
3. 必须和这个工具的功能真正相关，不要编造工具做不到的事
4. 只输出 JSON 数组，不要解释：["意图1", "意图2", ...]
"""


def synthesize_intents(
    tool_name: str,
    description: str,
    llm,
    n: int = _SYNTH_INTENTS_PER_TOOL,
) -> List[str]:
    """用 LLM 为一个 tool 合成 N 条口语意图。失败返回空列表。"""
    try:
        raw = llm.generate(
            [{"role": "user", "content": _SYNTH_PROMPT.format(
                n=n, tool_name=tool_name, description=description[:800],
            )}],
            mode="quick",
        )
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()][:n]
    except Exception:
        logger.exception("[mcp_intent] synthesize failed for %s", tool_name)
    return []


def _iter_server_tools(manager):
    """按 server 分组遍历 manager.tools（MCPToolSpec 字典）。"""
    by_server: Dict[str, List] = {}
    for spec in manager.tools.values():
        by_server.setdefault(spec.server_name, []).append(spec)
    for name, specs in by_server.items():
        yield name, specs


def build_index(servers: Dict[str, dict], manager, llm) -> Dict[str, Any]:
    """为所有已连接 server 的 tool 建意图索引。

    Args:
        servers: dynamic_store.load_all_servers() 的结果（当前未直接用，保留）
        manager: MCPClientManager
        llm: 有 .generate(messages, mode=...) 的 LLM 客户端

    Returns:
        {"indexed_tools": n, "indexed_intents": m}
    """
    import faiss
    import numpy as np

    encoder = get_encoder(_EMBED_MODEL, local_files_only=True)
    dim = encoder.get_sentence_embedding_dimension()

    records: List[Dict] = []
    tool_count = 0
    for server_name, specs in _iter_server_tools(manager):
        for spec in specs:
            tool_count += 1
            # 1. tool 原始描述（保证精确命中）
            records.append({
                "text": f"{spec.remote_name} {spec.description}",
                "server_name": server_name,
                "tool_name": spec.public_name,
                "kind": "description",
            })
            # 2. 合成意图
            intents = synthesize_intents(
                spec.remote_name, spec.description, llm,
            )
            for it in intents:
                records.append({
                    "text": it,
                    "server_name": server_name,
                    "tool_name": spec.public_name,
                    "kind": "intent",
                })

    if not records:
        return {"indexed_tools": 0, "indexed_intents": 0}

    os.makedirs(_INDEX_DIR, exist_ok=True)
    texts = [r["text"] for r in records]
    vecs = encoder.encode(
        texts, normalize_embeddings=True, show_progress_bar=False,
    )
    vecs = np.array(vecs, dtype="float32")

    index = faiss.IndexFlatIP(dim)   # 归一化后 IP = cosine
    index.add(vecs)
    faiss.write_index(index, _INDEX_FILE)
    with open(_RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info(
        "[mcp_intent] index built: %d tools, %d records",
        tool_count, len(records),
    )
    return {"indexed_tools": tool_count, "indexed_intents": len(records)}


def search_intents(
    query: str,
    top_k: int = 20,
    per_server_cap: int = 3,
    frame=None,
) -> List[Dict[str, Any]]:
    """混合检索意图索引，返回按 server 聚合的候选。

    步骤：
        1. QueryRewriter 改写 query（复用 rag/query_rewriter.py）
        2. 向量检索 top_k*3 + BM25 检索 top_k*3
        3. rrf_fuse 融合
        4. 按 server 聚合，每 server 保留最高分的 per_server_cap 条
    """
    import faiss
    import numpy as np
    from rag.bm25_store import BM25Store, rrf_fuse
    from rag.query_rewriter import get_query_rewriter

    if not (os.path.exists(_RECORDS_FILE) and os.path.exists(_INDEX_FILE)):
        return []

    with open(_RECORDS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        return []

    # 1. 改写
    try:
        rewritten = get_query_rewriter().rewrite(query)
    except Exception:
        logger.exception("[mcp_intent] rewrite failed, use raw query")
        rewritten = query

    # 2. 向量检索
    encoder = get_encoder(_EMBED_MODEL, local_files_only=True)
    qv = encoder.encode([rewritten], normalize_embeddings=True)[0]
    qv = np.array([qv], dtype="float32")
    index = faiss.read_index(_INDEX_FILE)
    k = min(top_k * 3, len(records))
    dists, idxs = index.search(qv, k)

    vec_results: List[Dict] = []
    for d, i in zip(dists[0], idxs[0]):
        if i < 0 or i >= len(records):
            continue
        r = dict(records[i])
        r["score"] = float(d)
        r["filename"] = r.get("server_name", "")
        r["metadata"] = {"chunk_id": f"{r['tool_name']}#vec-{i}"}
        vec_results.append(r)

    # 3. BM25 检索（BM25Store 从 metadata 构建，这里用 records 当 metadata）
    bm25_results: List[Dict] = []
    try:
        bm25 = BM25Store()
        bm25.build(records)
        bm25_results = bm25.search(rewritten, top_k * 3, lambda: records)
        for i, r in enumerate(bm25_results):
            r["metadata"] = {"chunk_id": f"{r.get('tool_name','')}#bm25-{i}"}
    except Exception:
        logger.exception("[mcp_intent] bm25 failed, vector only")

    # 4. RRF 融合
    if bm25_results:
        fused = rrf_fuse(vec_results, bm25_results, k=60, limit=top_k * 2)
    else:
        fused = vec_results[: top_k * 2]

    # 5. 领域过滤 + 按 server 聚合
    #   如果 frame 有 domain，记录 domain 不匹配的候选大幅降权，
    #   避免 filesystem 的"读取体育新闻文件"之类合成意图污染检索。
    by_server: Dict[str, List[Dict]] = {}
    for item in fused:
        srv = item.get("server_name") or ""
        if not srv:
            continue
        if frame is not None and frame.domain != "unknown":
            rec_domain = (item.get("intent_features") or {}).get("domain")
            if rec_domain and rec_domain != frame.domain:
                item["rrf_score"] = item.get("rrf_score", 0) * 0.2
        by_server.setdefault(srv, []).append(item)

    out: List[Dict[str, Any]] = []
    for srv, items in by_server.items():
        items.sort(key=lambda x: -x.get("rrf_score", 0))
        out.append({
            "server_name": srv,
            "top_score": items[0].get("rrf_score", 0),
            "matched_tools": items[:per_server_cap],
        })
    out.sort(key=lambda x: -x["top_score"])
    return out