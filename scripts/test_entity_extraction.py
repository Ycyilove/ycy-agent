#!/usr/bin/env python3
"""测试知识库文档的实体抽取齐全性与正确性（多 PDF 版）。"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_PDF_DIR = _ROOT / "data" / "compare_pdfs"
REPORT_NAME = _ROOT / "entity_report.json"

TEST_SCHEMA = {
    "types": {
        "person": {"header_hints": ["姓名", "名字", "队员", "选手", "name"], "is_entity": True, "is_anchor": True},
        "team": {"header_hints": ["团队", "队伍", "队名", "team"], "value_patterns": [r"[A-Za-z][A-Za-z0-9_\-]{2,}"], "is_entity": True, "is_anchor": True},
        "award": {"header_hints": ["奖项", "等级", "名次", "奖"], "is_entity": True, "is_anchor": True},
        "number": {"header_hints": ["分数", "得分", "成绩", "积分"], "is_entity": True},
        "school": {"header_hints": ["学校", "单位", "院校"], "is_entity": True},
    }
}

logging.getLogger("jieba").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

class _Prov:
    def __init__(self, page_no): self.page_no = page_no
class _FakeCell:
    def __init__(self, text): self.text = text
class _FakeGrid:
    def __init__(self, grid): self.grid = grid
class _FakeTable:
    def __init__(self, grid, pages):
        self.data = _FakeGrid([[_FakeCell(t) for t in row] for row in grid])
        self.prov = [_Prov(p) for p in pages]
class _FakeListItem:
    def __init__(self, text): self.text = text
class _FakeList:
    def __init__(self, items, pages):
        self.items = [_FakeListItem(t) for t in items]
        self.prov = [_Prov(p) for p in pages]
class _FakeDoc:
    def __init__(self, tables, lists):
        self.tables = tables
        self.lists = lists
class _FakeChunkMeta:
    def __init__(self, headings, pages, heading_levels=None):
        self.headings = headings
        self.page_numbers = pages
        self.heading_levels = heading_levels or []
class _FakeChunk:
    def __init__(self, text, headings, pages, heading_levels=None):
        self.text = text
        self.meta = _FakeChunkMeta(headings, pages, heading_levels)

def _cache_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_name(f"{pdf_path.stem}.parsed_cache.json")

def build_cache(pdf_path: Path, cache_path: Path) -> dict:
    from rag.docling_parser import DoclingParser
    parser = DoclingParser()
    chunks = parser.parse_to_chunks(str(pdf_path))
    doc = parser.last_document
    if doc is None:
        raise RuntimeError(f"[✗] Docling 解析失败，无法获取表格/列表结构: {pdf_path.name}")
        
    cache: dict = {"source": pdf_path.name, "chunks": [], "tables": [], "lists": []}
    for c in chunks:
        cache["chunks"].append({
            "text": getattr(c, "text", "") or "",
            "headings": list(getattr(c.meta, "headings", None) or []),
            "heading_levels": list(getattr(c.meta, "heading_levels", None) or []),
            "page_numbers": list(getattr(c.meta, "page_numbers", None) or []),
        })
    if doc is not None:
        for t in getattr(doc, "tables", []) or []:
            try:
                grid = [[(getattr(cell, "text", "") or "") for cell in row] for row in t.data.grid]
            except AttributeError: continue
            pages = [getattr(p, "page_no", None) for p in (getattr(t, "prov", None) or [])]
            cache["tables"].append({"grid": grid, "page_numbers": [p for p in pages if p]})
        for lst in getattr(doc, "lists", []) or []:
            try:
                items = [(getattr(it, "text", "") or "") for it in lst.items]
            except AttributeError: continue
            pages = [getattr(p, "page_no", None) for p in (getattr(lst, "prov", None) or [])]
            cache["lists"].append({"items": items, "page_numbers": [p for p in pages if p]})
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache

def load_cache(cache_path: Path) -> dict:
    return json.loads(cache_path.read_text(encoding="utf-8"))

def rebuild_from_cache(cache: dict):
    doc = _FakeDoc(
        tables=[_FakeTable(t["grid"], t.get("page_numbers", [])) for t in cache.get("tables", [])],
        lists=[_FakeList(l["items"], l.get("page_numbers", [])) for l in cache.get("lists", [])],
    )
    chunks = [
        _FakeChunk(c["text"], c.get("headings", []), c.get("page_numbers", []), c.get("heading_levels", []))
        for c in cache.get("chunks", [])
    ]
    return doc, chunks

def run_extraction(doc, chunks, filename: str, schema, extractor) -> tuple:
    from rag.structure.parser import parse_structure
    records = parse_structure(doc, chunks, filename, schema=schema)
    mentions: List = []
    cell_mentions: Dict[tuple, List] = {}
    for rec in records:
        for cell in rec.cells:
            ms = extractor.extract(cell.text, chunk_id=rec.record_id)
            cell_mentions[(rec.record_id, cell.col_index)] = ms
            mentions.extend(ms)
    for ch in chunks:
        mentions.extend(extractor.extract(ch.text))
    return records, mentions, cell_mentions

def _value_matchable(schema, field_type: str, text: str) -> bool:
    td = schema.types.get(field_type)
    if td is None: return False
    if text in td.dictionary: return True
    for p in td.value_patterns:
        try:
            if re.fullmatch(p, text) or re.search(p, text): return True
        except re.error: continue
    return False

def build_report(name: str, records, mentions, cell_mentions, schema) -> dict:
    from rag.entity.normalizer import normalize_entity
    declared = {t for t, td in schema.types.items() if td.is_entity}
    by_kind: Dict[str, set] = defaultdict(set)
    for m in mentions: by_kind[m.kind].add(m.surface)
    missing, header_only = [], []
    for rec in records:
        for cell in rec.cells:
            if cell.field_type not in declared: continue
            ms = cell_mentions.get((rec.record_id, cell.col_index), [])
            key = normalize_entity(cell.text)
            if any(m.normalized == key and m.kind == cell.field_type for m in ms): continue
            entry = {"record_id": rec.record_id, "col_name": cell.col_name, "text": cell.text, "field_type": cell.field_type, "type_source": cell.type_source, "mentions": [f"{m.surface}/{m.kind}" for m in ms]}
            if _value_matchable(schema, cell.field_type, cell.text): missing.append(entry)
            else: header_only.append(entry)
    return {"name": name, "records": len(records), "mentions": len(mentions), "by_kind": {k: sorted(v) for k, v in by_kind.items()}, "missing": missing, "header_only": header_only}

def print_report(pdf_name: str, r: dict) -> None:
    print(f"\n===== {pdf_name} :: {r['name']} =====")
    print(f"record 数: {r['records']} | mention 总数: {r['mentions']}")
    for kind in sorted(r["by_kind"]):
        items = r["by_kind"][kind]
        print(f"  {kind}: {len(items)} 个 (示例: {', '.join(items[:5])})")
    print(f"覆盖率 missing: {len(r['missing'])} | 仅靠列名: {len(r['header_only'])}")

def _process_one_pdf(pdf_path: Path, rebuild_cache: bool, default_schema, cfg_schema) -> dict:
    from rag.entity.extractor import EntityExtractor
    cache_path = _cache_path_for(pdf_path)
    if cache_path.exists() and not rebuild_cache:
        print(f"    读取缓存: {cache_path.name}")
        cache = load_cache(cache_path)
    else:
        print(f"    解析 PDF（耗时较长）: {pdf_path.name}")
        cache = build_cache(pdf_path, cache_path)
        print(f"    已缓存: {cache_path.name}")
    doc, chunks = rebuild_from_cache(cache)
    print(f"    重建: chunks={len(chunks)}, tables={len(doc.tables)}, lists={len(doc.lists)}")
    
    default_ext = EntityExtractor()
    records_d, mentions_d, cell_mentions_d = run_extraction(doc, chunks, pdf_path.name, default_schema, default_ext)
    report_d = build_report("default", records_d, mentions_d, cell_mentions_d, default_schema)
    
    cfg_ext = EntityExtractor.from_schema(cfg_schema)
    records_c, mentions_c, cell_mentions_c = run_extraction(doc, chunks, pdf_path.name, cfg_schema, cfg_ext)
    report_c = build_report("configured", records_c, mentions_c, cell_mentions_c, cfg_schema)
    
    print_report(pdf_path.name, report_d)
    print_report(pdf_path.name, report_c)
    return {"default": report_d, "configured": report_c}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-cache", action="store_true", help="强制重新解析")
    ap.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="PDF 目录")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists(): sys.exit(f"[✗] PDF 目录不存在: {pdf_dir}")
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths: sys.exit(f"[✗] {pdf_dir} 下没有 PDF")
    
    print(f"[0] 发现 {len(pdf_paths)} 个 PDF")
    from rag.structure.field_classifier import FieldSchema
    default_schema = FieldSchema.empty()
    cfg_schema = FieldSchema.from_dict(TEST_SCHEMA)
    
    all_reports = {}
    for pdf_path in pdf_paths:
        print(f"\n[+] 处理: {pdf_path.name}")
        all_reports[pdf_path.name] = _process_one_pdf(pdf_path, args.rebuild_cache, default_schema, cfg_schema)
        
    REPORT_NAME.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] 完整实体清单已保存: {REPORT_NAME.name}")

if __name__ == "__main__":
    main()