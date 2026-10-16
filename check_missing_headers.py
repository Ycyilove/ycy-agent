# check_missing_headers.py
import os, sys, logging
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for n in ["docling", "httpx", "transformers", "RapidOCR", "jieba"]:
    logging.getLogger(n).setLevel(logging.ERROR)

pdf = next((str(p) for p in Path(".").rglob("*.pdf") if "天梯" in p.name), None)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import SectionHeaderItem

opts = PdfPipelineOptions()
opts.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
doc = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}).convert(pdf).document

# 遍历，找"无内容标题"
items_seq = []
for idx, (item, level) in enumerate(doc.iterate_items()):
    items_seq.append((item, level, idx))

for i, (item, level, idx) in enumerate(items_seq):
    if not isinstance(item, SectionHeaderItem):
        continue
    text = (getattr(item, "text", "") or "").strip()
    if "全国" not in text and "团队" not in text:
        continue
    # 看它后面第一个非标题 item
    has_content = False
    for j in range(i + 1, len(items_seq)):
        nitem, nlevel, nidx = items_seq[j]
        if not isinstance(nitem, SectionHeaderItem):
            has_content = True
            break
        if nlevel > level:
            continue
        break
    print(f"idx={idx} level={level} has_content={has_content} text={text!r}")