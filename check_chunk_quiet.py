"""静音版 chunk 诊断：只搜'团队'和'全国'，强制 UTF-8 输出。"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import logging
from pathlib import Path

# ── 强制 stdout UTF-8，避免 GBK 崩溃 ──
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

for name in ["docling", "httpx", "transformers", "RapidOCR", "jieba", "faiss"]:
    logging.getLogger(name).setLevel(logging.ERROR)

pdf_path = next(
    (str(p) for p in Path(".").rglob("*.pdf") if "天梯" in p.name or "获奖" in p.name),
    None,
)
if not pdf_path:
    print("no pdf")
    sys.exit(1)

from rag.docling_parser import DoclingParser

parser = DoclingParser()
chunks = parser.parse_to_chunks(pdf_path)
print(f"total: {len(chunks)}")


def get_pages(dc):
    pages = set()
    for item in (getattr(dc.meta, "doc_items", None) or []):
        for prov in (getattr(item, "prov", None) or []):
            p = getattr(prov, "page_no", None)
            if p is not None:
                pages.add(p)
    return sorted(pages)


# ── 1. 搜"团队" ──
print("=" * 60)
print("[1] '团队' hits")
print("=" * 60)
hits = 0
for i, dc in enumerate(chunks):
    text = dc.text or ""
    headings = " | ".join(getattr(dc.meta, "headings", None) or [])
    if "团队" in headings or "团队" in text:
        hits += 1
        print(f"  chunk {i} pages={get_pages(dc)}")
        print(f"    headings: {headings!r}")
        print(f"    text[:150]: {text[:150]!r}")
        if hits >= 20:
            print("  ... (只显示前 20)")
            break
if hits == 0:
    print("  (无)")

# ── 2. 搜"团" 单独看（OCR 可能把"团"识别错）──
print()
print("=" * 60)
print("[2] '团' 单独搜")
print("=" * 60)
hits = 0
for i, dc in enumerate(chunks):
    text = dc.text or ""
    headings = " | ".join(getattr(dc.meta, "headings", None) or [])
    if "团" in headings or "团" in text:
        hits += 1
        print(f"  chunk {i} pages={get_pages(dc)}")
        print(f"    headings: {headings!r}")
        print(f"    text[:120]: {text[:120]!r}")
        if hits >= 30:
            print("  ... (只显示前 30)")
            break
if hits == 0:
    print("  (无)")

# ── 3. 搜"二等" ──
print()
print("=" * 60)
print("[3] '二等' hits")
print("=" * 60)
hits = 0
for i, dc in enumerate(chunks):
    text = dc.text or ""
    headings = " | ".join(getattr(dc.meta, "headings", None) or [])
    if "二等" in headings or "二等" in text:
        hits += 1
        print(f"  chunk {i} pages={get_pages(dc)}")
        print(f"    headings: {headings!r}")
        print(f"    text[:120]: {text[:120]!r}")
        if hits >= 30:
            print("  ... (只显示前 30)")
            break
if hits == 0:
    print("  (无)")

# ── 4. 第 1~15 页所有 chunk 的 heading ──
print()
print("=" * 60)
print("[4] 第 1~15 页所有 chunk 的 heading")
print("=" * 60)
for i, dc in enumerate(chunks):
    pages = get_pages(dc)
    if 21 in pages:
        print(f"--- chunk {i} pages={pages} ---")
        print(f"headings: {getattr(dc.meta, 'headings', None)}")
        print(f"FULL TEXT:\n{dc.text}")
        print()
    if any(1 <= p <= 15 for p in pages):
        headings = getattr(dc.meta, "headings", None) or []
        text = (dc.text or "")[:60]
        print(f"  chunk {i} pages={pages} headings={headings!r}")
        print(f"    text: {text!r}")

# ── 5. 第 20~22 页的完整 text ──
print()
print("=" * 60)
print("[5] 第 20~22 页完整 text")
print("=" * 60)
for i, dc in enumerate(chunks):
    pages = get_pages(dc)
    if any(20 <= p <= 22 for p in pages):
        headings = getattr(dc.meta, "headings", None) or []
        text = dc.text or ""
        print(f"--- chunk {i} pages={pages} ---")
        print(f"headings: {headings!r}")
        print(f"FULL TEXT:")
        print(text)
        print()