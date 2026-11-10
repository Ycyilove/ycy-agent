"""记录级索引 + 实体倒排。

设计：
    - records.json：全量记录（原子写）
    - 内存 BM25 索引，随记录变更 reset
    - 实体倒排 normalized -> {field_type -> [record_id]}
    - 哪些字段类型进倒排由 FieldSchema 决定，不硬编码领域类型
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

from rag.entity.normalizer import normalize_entity
from rag.structure.field_classifier import FieldSchema, get_default_schema
from rag.structure.types import BlockKind, Record

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    try:
        import jieba
        return [t for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        return list(text)


class RecordStore:
    def __init__(
        self,
        path: str = "./vector_store/records.json",
        schema: Optional[FieldSchema] = None,
    ):
        self._schema = schema or get_default_schema()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: Dict[str, Record] = self._load()
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._inverted: Dict[str, Dict[str, List[str]]] = {}
        self._rebuild_indexes()

    # ── 持久化 ──

    def _load(self) -> Dict[str, Record]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            logger.exception("[dsh][record] 加载失败，从空开始")
            return {}
        out: Dict[str, Record] = {}
        items = raw if isinstance(raw, list) else raw.values()
        for item in items:
            try:
                record = Record.from_dict(item)
                out[record.record_id] = record
            except (KeyError, TypeError):
                continue
        return out

    def _persist(self) -> None:
        fd, tmp = tempfile.mkstemp(
            prefix=".records.", suffix=".tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    [r.to_dict() for r in self._records.values()],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            os.replace(tmp, self._path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ── 索引 ──

    def _rebuild_indexes(self) -> None:
        with self._lock:
            self._inverted = {}
            self._bm25_ids = list(self._records.keys())
            self._bm25 = None
            for rid, record in self._records.items():
                for cell in record.cells:
                    if not cell.text.strip():
                        continue
                    if not self._schema.is_entity_type(cell.field_type):
                        continue
                    key = normalize_entity(cell.text)
                    if not key:
                        continue
                    self._inverted.setdefault(key, {}).setdefault(
                        cell.field_type, []
                    ).append(rid)

    def _ensure_bm25(self) -> bool:
        if self._bm25 is not None:
            return True
        if not self._records:
            return False
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.debug("[dsh][record] rank_bm25 未安装，跳过记录级 BM25")
            return False
        with self._lock:
            corpus = [
                _tokenize(self._records[rid].to_index_text())
                for rid in self._bm25_ids
            ]
            self._bm25 = BM25Okapi(corpus)
        return True

    # ── 写入 / 删除 ──

    def add(self, records: List[Record]) -> int:
        if not records:
            return 0
        with self._lock:
            for r in records:
                self._records[r.record_id] = r
            self._rebuild_indexes()
            self._persist()
        logger.info(
            "[dsh][record] 新增 %d 条，总计 %d 条",
            len(records), len(self._records),
        )
        return len(records)

    def delete_by_source(self, filename: str) -> int:
        with self._lock:
            ids = [
                rid for rid, r in self._records.items()
                if r.source_filename == filename
            ]
            for rid in ids:
                self._records.pop(rid, None)
            if ids:
                self._rebuild_indexes()
                self._persist()
        return len(ids)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._rebuild_indexes()
            self._persist()

    # ── 查询 ──

    def search(self, query: str, top_k: int = 20) -> List[Record]:
        if not self._ensure_bm25():
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        with self._lock:
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
            return [
                self._records[self._bm25_ids[i]]
                for i, _ in ranked
                if i < len(self._bm25_ids)
            ]

    def by_entity(
        self, normalized: str, kind: Optional[str] = None
    ) -> List[Record]:
        with self._lock:
            kinds = self._inverted.get(normalized, {})
            ids: List[str] = []
            if kind:
                ids = list(kinds.get(kind, []))
            else:
                for v in kinds.values():
                    ids.extend(v)
            seen, out = set(), []
            for rid in ids:
                if rid in seen or rid not in self._records:
                    continue
                seen.add(rid)
                out.append(self._records[rid])
            return out

    def get(self, record_id: str) -> Optional[Record]:
        with self._lock:
            return self._records.get(record_id)

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_records": len(self._records),
                "by_block": {
                    k.value: sum(
                        1 for r in self._records.values()
                        if r.block_kind == k
                    )
                    for k in BlockKind
                },
                "entities": len(self._inverted),
            }