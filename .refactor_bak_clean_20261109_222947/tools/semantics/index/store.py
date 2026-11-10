"""FAISS + records.json 的统一读写。

设计：
    - 原子写：先写临时文件，再 os.replace
    - records 里带 vector 字段，启动时零编码
    - fingerprint 包含 text + chunk_id，避免缓存错位
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def fingerprint(records: List[Dict[str, Any]]) -> str:
    """对 records 的 tool_name + text 做指纹，用于缓存失效判断。"""
    h = hashlib.md5()
    for r in records:
        h.update((r.get("tool_name") or "").encode("utf-8", "ignore"))
        h.update(b"\x00")
        h.update((r.get("text") or "").encode("utf-8", "ignore"))
        h.update(b"\x01")
    return h.hexdigest()


def write_index_atomic(
    records: List[Dict[str, Any]],
    index_dir: str,
) -> None:
    """原子写 records.json + faiss.index。

    records 里必须有 vector 字段（list[float]）。
    """
    if not records:
        logger.info("[store] no records, skip write")
        return

    os.makedirs(index_dir, exist_ok=True)
    records_file = os.path.join(index_dir, "records.json")
    index_file = os.path.join(index_dir, "faiss.index")

    import faiss
    import numpy as np

    vectors = [r["vector"] for r in records]
    arr = np.array(vectors, dtype="float32")
    dim = arr.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(arr)

    fd, tmp_records = tempfile.mkstemp(
        dir=index_dir, suffix=".records.json.tmp",
    )
    os.close(fd)
    with open(tmp_records, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    fd, tmp_index = tempfile.mkstemp(
        dir=index_dir, suffix=".faiss.index.tmp",
    )
    os.close(fd)
    faiss.write_index(index, tmp_index)

    os.replace(tmp_records, records_file)
    os.replace(tmp_index, index_file)
    logger.info(
        "[store] index written: %d records, dim=%d",
        len(records), dim,
    )


def load_records(index_dir: str) -> List[Dict[str, Any]]:
    """读 records.json。缺失/损坏返回 []。"""
    records_file = os.path.join(index_dir, "records.json")
    if not os.path.exists(records_file):
        return []
    try:
        with open(records_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        logger.exception("[store] load records failed")
        return []


def load_index(index_dir: str):
    """读 faiss.index。缺失返回 None。"""
    index_file = os.path.join(index_dir, "faiss.index")
    if not os.path.exists(index_file):
        return None
    try:
        import faiss
        return faiss.read_index(index_file)
    except Exception:
        logger.exception("[store] load index failed")
        return None