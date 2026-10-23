"""离线建意图索引（只覆盖已配置 server，不联网 Registry）。

用法：
    # 全量（首次或手动）
    python scripts/build_capability_index.py

    # 增量（应用内调用，只处理指定 server）
    from scripts.build_capability_index import build_incremental
    await build_incremental(manager, llm, {"new_server"})

数据来源：
    - manager.tools（已连接 server 注册的工具）
    - manager.configs（已配置 server 的 MCPServerConfig）

不联网 Registry。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from infra.offline import configure_hf_env
configure_hf_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_INDEX_DIR = os.path.join(_PROJECT_ROOT, "data", "capability_index")
_RECORDS_FILE = os.path.join(_INDEX_DIR, "records.json")
_INDEX_FILE = os.path.join(_INDEX_DIR, "faiss.index")

_SYNTH_N = 8
_BATCH_SIZE = 5          # 每批 5 个 tool 1 次 LLM 调用
_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# ── 批量合成 prompt ──
_BATCH_PROMPT = """给定以下 {n} 个 MCP 工具，为每个生成 {k} 条中文口语化用户意图，并标注结构化特征。

{tools_block}

输出严格 JSON 数组，长度必须等于 {n}：
[
  {{
    "tool_name": "工具名（原样）",
    "intents": ["意图1", "意图2", ...],
    "features": {{
      "domain": "news|sports|knowledge|file|code|web|unknown",
      "action": "search|read|write|compute|browse",
      "freshness": "realtime|recent|any"
    }}
  }},
  ...
]

要求：
- 意图是用户可能说的话，口语化，覆盖不同表达
- 意图必须和工具功能真正相关
- features 取整组意图的共性
- 只输出 JSON，不要解释
"""


# ─────────────────────────────────────────────────────────────
# 合成意图（批量 + 降级）
# ─────────────────────────────────────────────────────────────

async def _synth_batch_llm(batch: List[Dict[str, str]], llm) -> Dict[str, Dict]:
    """批量合成。返回 {tool_name: {"intents": [...], "features": {...}}}。

    批量失败时返回空 dict，由调用方降级到逐 tool。
    """
    tools_block = "\n".join(
        f"{i+1}. tool_name={t['remote_name']}\n   description={t['description'][:300]}"
        for i, t in enumerate(batch)
    )
    prompt = _BATCH_PROMPT.format(
        n=len(batch), k=_SYNTH_N, tools_block=tools_block,
    )
    try:
        raw = await asyncio.to_thread(
            llm.generate,
            [{"role": "user", "content": prompt}],
            mode="quick",
        )
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"expected list, got {type(data).__name__}")
        result = {}
        for item in data:
            name = item.get("tool_name")
            if name:
                result[name] = {
                    "intents": item.get("intents", [])[:_SYNTH_N],
                    "features": item.get("features", {}),
                }
        return result
    except Exception:
        logger.warning(
            "[build] batch synth failed (n=%d), will fallback", len(batch),
        )
        return {}


async def _synth_single_llm(tool: Dict[str, str], llm) -> Dict:
    """单 tool 合成（批量失败时的降级）。"""
    prompt = (
        f"给定一个 MCP 工具，生成 {_SYNTH_N} 条中文口语化用户意图，并标注结构化特征。\n\n"
        f"工具名：{tool['remote_name']}\n"
        f"工具描述：{tool['description'][:500]}\n\n"
        "输出严格 JSON：\n"
        '{"intents": ["意图1", ...], "features": {"domain": "...", '
        '"action": "...", "freshness": "..."}}\n\n'
        "只输出 JSON，不要解释"
    )
    try:
        raw = await asyncio.to_thread(
            llm.generate,
            [{"role": "user", "content": prompt}],
            mode="quick",
        )
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        data = json.loads(raw)
        return {
            "intents": data.get("intents", [])[:_SYNTH_N],
            "features": data.get("features", {}),
        }
    except Exception:
        logger.exception("[build] single synth failed for %s", tool["remote_name"])
        return {"intents": [], "features": {}}


async def _synth_all(tools: List[Dict[str, str]], llm) -> Dict[str, Dict]:
    """批量合成所有 tool，批量失败的 batch 降级到逐 tool。"""
    result: Dict[str, Dict] = {}
    for i in range(0, len(tools), _BATCH_SIZE):
        batch = tools[i:i + _BATCH_SIZE]
        logger.info(
            "[build] batch %d/%d: %d tools",
            i // _BATCH_SIZE + 1,
            (len(tools) + _BATCH_SIZE - 1) // _BATCH_SIZE,
            len(batch),
        )
        batch_result = await _synth_batch_llm(batch, llm)
        # 批量里没返回的 tool → 逐 tool 降级
        missing = [
            t for t in batch if t["remote_name"] not in batch_result
        ]
        for t in missing:
            logger.info(
                "[build] fallback single synth: %s", t["remote_name"],
            )
            batch_result[t["remote_name"]] = await _synth_single_llm(t, llm)
        result.update(batch_result)
    return result


# ─────────────────────────────────────────────────────────────
# 记录构造 / 读写
# ─────────────────────────────────────────────────────────────

def _server_tool_specs(manager, server_names: Optional[Set[str]] = None) -> List[Dict[str, str]]:
    """从 manager 拿 tool 列表。server_names 非 None 时只保留这些 server。"""
    out = []
    for spec in list(manager.tools.values()):
        if server_names is not None and spec.server_name not in server_names:
            continue
        out.append({
            "server_name": spec.server_name,
            "remote_name": spec.remote_name,
            "public_name": spec.public_name,
            "description": spec.description or "",
        })
    return out


def _build_records(
    specs: List[Dict[str, str]], synth: Dict[str, Dict],
) -> List[Dict]:
    records = []
    for spec in specs:
        name = spec["public_name"]
        server_name = spec["server_name"]
        remote_name = spec["remote_name"]
        desc = spec["description"]

        injection_suspected = any(
            k in desc.lower()
            for k in ("ignore previous", "system:", "<|im_start|>")
        )
        safe_desc = desc[:200] if injection_suspected else desc[:800]

        syn = synth.get(remote_name, {})
        intents = syn.get("intents", [])
        features = syn.get("features", {})

        records.append({
            "record_id": f"{name}#desc",
            "server_name": server_name,
            "tool_name": name,
            "kind": "tool_desc",
            "text": f"{remote_name} {safe_desc[:500]}",
            "intent_features": features,
            "provenance": {"injection_suspected": injection_suspected},
        })
        for i, it in enumerate(intents):
            records.append({
                "record_id": f"{name}#intent-{i}",
                "server_name": server_name,
                "tool_name": name,
                "kind": "synthetic_intent",
                "text": it,
                "intent_features": features,
                "provenance": {"injection_suspected": injection_suspected},
            })
    return records


def _load_existing_records() -> List[Dict]:
    if not os.path.exists(_RECORDS_FILE):
        return []
    try:
        with open(_RECORDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("[build] load existing records failed")
        return []


async def _write_index(records: List[Dict]) -> None:
    """原子写入 records.json + faiss.index。"""
    if not records:
        logger.info("[build] no records, skip write")
        return
    os.makedirs(_INDEX_DIR, exist_ok=True)

    from infra.embedding import get_encoder
    enc = get_encoder(_EMBED_MODEL, local_files_only=True)
    dim = enc.get_sentence_embedding_dimension()

    texts = [r["text"] for r in records]
    vecs = await asyncio.to_thread(
        enc.encode, texts,
        normalize_embeddings=True, show_progress_bar=False,
    )

    import faiss
    import numpy as np
    arr = np.array(vecs, dtype="float32")
    index = faiss.IndexFlatIP(dim)
    index.add(arr)

    # 原子写
    fd, tmp_records = tempfile.mkstemp(dir=_INDEX_DIR, suffix=".records.json.tmp")
    os.close(fd)
    with open(tmp_records, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    fd, tmp_index = tempfile.mkstemp(dir=_INDEX_DIR, suffix=".faiss.index.tmp")
    os.close(fd)
    await asyncio.to_thread(faiss.write_index, index, tmp_index)

    os.replace(tmp_records, _RECORDS_FILE)
    os.replace(tmp_index, _INDEX_FILE)
    logger.info(
        "[build] index written: %d records, dim=%d", len(records), dim,
    )


# ─────────────────────────────────────────────────────────────
# 公开入口
# ─────────────────────────────────────────────────────────────

async def build_full(manager, llm) -> int:
    """全量重建。返回 record 数。"""
    specs = _server_tool_specs(manager)
    if not specs:
        logger.info("[build] no MCP tools registered, skip")
        return 0
    logger.info("[build] full rebuild: %d tools", len(specs))
    synth = await _synth_all(specs, llm)
    records = _build_records(specs, synth)
    await _write_index(records)
    logger.info("[build] full done: %d tools, %d records", len(specs), len(records))
    return len(records)


async def build_incremental(manager, llm, server_names: Set[str]) -> int:
    """增量重建：只处理 server_names 里的 server。

    1. 读现有 records
    2. 删除 server_names 里的旧记录
    3. 对新 tool 批量合成
    4. 追加新记录
    5. 重建 FAISS
    """
    if not server_names:
        return 0

    # 现有记录里，删除目标 server 的旧记录
    existing = _load_existing_records()
    kept = [
        r for r in existing
        if r.get("server_name") not in server_names
    ]
    removed = len(existing) - len(kept)

    # 目标 server 的当前 tool
    specs = _server_tool_specs(manager, server_names=server_names)
    if not specs:
        logger.info(
            "[build] incremental: no tools for %r (removed %d old records)",
            server_names, removed,
        )
        await _write_index(kept)
        return len(kept)

    logger.info(
        "[build] incremental: %d tools, removed %d old records",
        len(specs), removed,
    )
    synth = await _synth_all(specs, llm)
    new_records = _build_records(specs, synth)
    all_records = kept + new_records
    await _write_index(all_records)
    logger.info(
        "[build] incremental done: +%d records (total %d)",
        len(new_records), len(all_records),
    )
    return len(all_records)


# ─────────────────────────────────────────────────────────────
# 手动跑入口（自己构造临时 manager）
# ─────────────────────────────────────────────────────────────

async def _build_with_temp_manager(llm) -> int:
    """手动跑时：自己构造临时 manager，连上 mcp_servers.json 里的 server。"""
    from mcp_client import MCPClientManager, load_mcp_config

    configs = load_mcp_config()
    if not configs:
        logger.info("[build] mcp_servers.json is empty, skip")
        return 0

    manager = MCPClientManager(configs)

    def _noop_register(**kwargs):
        return None

    def _noop_unregister(name):
        return None

    try:
        for name, config in manager.configs.items():
            if not getattr(config, "enabled", True):
                logger.info("[build] skip disabled server: %s", name)
                continue

            # stdio 命令预检：命令不存在直接跳过，不重试
            if config.transport == "stdio":
                from backend.routes.mcp import _resolve_command
                if _resolve_command(config.command) is None:
                    logger.warning(
                        "[build] skip %s: command not found (%s)",
                        name, config.command,
                    )
                    continue

            try:
                await asyncio.wait_for(manager.connect(name), timeout=20.0)
                await asyncio.wait_for(
                    manager.discover(name, _noop_register, _noop_unregister),
                    timeout=20.0,
                )
                logger.info("[build] connected: %s", name)
            except Exception as exc:
                logger.warning(
                    "[build] connect %s failed, skip: %s: %s",
                    name, type(exc).__name__, exc,
                )

        return await build_full(manager, llm)
    finally:
        try:
            await manager.close()
        except Exception:
            logger.exception("[build] manager.close failed")


def main():
    """手动跑入口。"""
    from backend.app import get_llm
    llm = get_llm()
    logger.info("[build] manager is None, building with temp manager")
    asyncio.run(_build_with_temp_manager(llm))


if __name__ == "__main__":
    main()