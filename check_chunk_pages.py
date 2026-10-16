# check_chunk_pages.py
import os, sys, logging
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

for n in ["docling", "httpx", "transformers", "RapidOCR", "jieba"]:
    logging.getLogger(n).setLevel(logging.ERROR)

pdf = next((str(p) for p in Path(".").rglob("*.pdf") if "天梯" in p.name), None)
print(f"PDF: {pdf}")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker

opts = PdfPipelineOptions()
opts.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
doc = converter.convert(pdf).document

chunker = HybridChunker()
chunks = list(chunker.chunk(dl_doc=doc))

print(f"total: {len(chunks)}")
for c in chunks[:3]:
    meta = c.meta
    print(f"--- {c.text[:40]!r} ---")
    print(f"  meta fields: {list(meta.__dict__.keys())}")
    print(f"  page_numbers: {getattr(meta, 'page_numbers', None)}")
    for it in (getattr(meta, "doc_items", None) or [])[:1]:
        print(f"  item type: {type(it).__name__}")
        print(f"  item dict: {list(it.__dict__.keys())}")
        prov = getattr(it, "prov", None)
        print(f"  prov: {prov}")
        if prov:
            for p in prov[:2]:
                print(f"    page_no: {getattr(p, 'page_no', None)}")