# -*- coding: utf-8 -*-
"""验证 self_ref 匹配。"""
import os
import sys
import logging
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for n in ["docling", "httpx", "transformers", "RapidOCR", "jieba"]:
    logging.getLogger(n).setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

pdf = None
for p in Path(".").rglob("*.pdf"):
    if "天梯" in p.name or "获奖" in p.name:
        pdf = str(p)
        break
if pdf is None:
    print("❌ 找不到 PDF")
    sys.exit(1)

print(f"PDF: {pdf}\n")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling_core.types.doc import SectionHeaderItem

opts = PdfPipelineOptions()
opts.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
)
print("解析中...")
doc = converter.convert(pdf).document
print("解析完成\n")

# ── 1. doc.iterate_items() 收集所有 SectionHeaderItem 的 self_ref ──
print("=" * 70)
print("[1] doc.iterate_items() 里所有 SectionHeaderItem")
print("=" * 70)

doc_header_refs = {}   # self_ref -> (text, page_no)
doc_all_refs = {}      # self_ref -> (type, text, page_no)

for item, level in doc.iterate_items():
    sr = getattr(item, "self_ref", None)
    text = (getattr(item, "text", "") or "").strip()
    prov = getattr(item, "prov", None) or []
    pages = [getattr(p, "page_no", None) for p in prov]
    page_no = pages[0] if pages else None

    if sr:
        doc_all_refs[sr] = (type(item).__name__, text[:50], page_no)

    if isinstance(item, SectionHeaderItem) and sr:
        doc_header_refs[sr] = (text, page_no)
        if "团队" in text or "全国" in text:
            print(f"  self_ref={sr!r} page={page_no} text={text!r}")

print(f"\n  共 {len(doc_all_refs)} 个 item，{len(doc_header_refs)} 个 SectionHeaderItem")

# ── 2. chunk 的 doc_items[].self_ref ──
print()
print("=" * 70)
print("[2] chunk 的 doc_items[].self_ref 匹配")
print("=" * 70)

chunker = HybridChunker()
chunks = list(chunker.chunk(dl_doc=doc))
print(f"total chunks: {len(chunks)}\n")

match_count = 0
mismatch_count = 0
sample_printed = 0

for i, c in enumerate(chunks):
    items = getattr(c.meta, "doc_items", None) or []
    if not items:
        continue

    first = items[0]
    sr = getattr(first, "self_ref", None)
    is_match = sr in doc_all_refs

    if is_match:
        match_count += 1
    else:
        mismatch_count += 1

    # 只打印前几个 + 含"团队"的
    text = (c.text or "")[:50].replace("\n", " ")
    if sample_printed < 5 or "团队" in (c.text or ""):
        doc_info = doc_all_refs.get(sr, ("?", "?", "?"))
        mark = "✅" if is_match else "❌"
        print(f"--- chunk {i} text={text!r} ---")
        print(f"  headings={getattr(c.meta, 'headings', None)}")
        print(f"  first doc_item self_ref={sr!r} {mark}")
        print(f"    doc[{sr}] = type={doc_info[0]} text={doc_info[1]!r} page={doc_info[2]}")
        sample_printed += 1

print()
print(f"匹配统计: ✅ {match_count} / ❌ {mismatch_count}")