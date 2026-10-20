"""记录级检索：记录召回 + 有限下钻。

不做全图共现：只在命中记录的字段值上做实体倒排下钻，
边数从 O(n^2) 降到 O(命中记录 × 字段数)。
"""

from __future__ import annotations

import logging
from typing import Dict, List

from rag.entity.extractor import EntityExtractor
from rag.structure.field_classifier import FieldSchema, get_default_schema
from rag.structure.types import Record

logger = logging.getLogger(__name__)


def record_search(
    query: str,
    record_store,
    extractor: EntityExtractor,
    top_k: int = 10,
    max_expand: int = 3,
    schema: Optional[FieldSchema] = None,
) -> dict:
    if schema is None:
        schema = get_default_schema()
    anchor_kinds = schema.anchor_types()
    """返回 {records, paths, warnings}。"""
    warnings: List[dict] = []
    mentions = extractor.extract(query)
    if anchor_kinds:
        anchors = [m for m in mentions if m.kind in anchor_kinds]
    else:
        # schema 未声明锚点类型时，用抽取器给出的非语法类别作为锚点
        anchors = [m for m in mentions if m.kind not in ("number", "date")]

    # 1. 记录级召回
    base = record_store.search(query, top_k=top_k * 2)

    # 2. 实体下钻：锚点实体命中的记录
    expanded: Dict[str, Record] = {r.record_id: r for r in base}
    paths: List[dict] = []
    for m in anchors:
        hits = record_store.by_entity(m.normalized, kind=m.kind)
        if not hits:
            warnings.append({"type": "entity_not_found", "entity": m.surface, "kind": m.kind})
            continue
        for rec in hits[:max_expand]:
            expanded.setdefault(rec.record_id, rec)
            if rec.record_id not in {r.record_id for r in base}:
                paths.append({
                    "from_entity": m.surface,
                    "record_id": rec.record_id,
                    "text": rec.to_index_text(),
                    "conf": 0.85,
                })

    # 3. 同记录内两跳：锚点记录的字段值再下钻一次
    for rec in list(expanded.values()):
        for (kind_a, val_a), (kind_b, val_b) in rec.entity_pairs():
            # entity_pairs 已排除 free_text，这里只保留 schema 认可的实体类型
            if not schema.is_entity_type(kind_a) or not schema.is_entity_type(kind_b):
                continue
            for m in anchors:
                if m.normalized not in {val_a, val_b}:
                    continue
                other = val_b if m.normalized == val_a else val_a
                from rag.entity.normalizer import normalize_entity
                next_hits = record_store.by_entity(normalize_entity(other))
                for nxt in next_hits[:max_expand]:
                    if nxt.record_id in expanded:
                        continue
                    expanded[nxt.record_id] = nxt
                    paths.append({
                        "from_entity": m.surface,
                        "via": other,
                        "record_id": nxt.record_id,
                        "text": nxt.to_index_text(),
                        "conf": 0.70,
                    })

    records = list(expanded.values())[:top_k]
    if not records:
        warnings.append({"type": "no_record_hit", "query": query})

    return {"records": records, "paths": paths, "warnings": warnings}