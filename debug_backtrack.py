# debug_backtrack.py
import os, sys, time, logging
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for n in ["docling", "httpx", "transformers", "RapidOCR", "jieba"]:
    logging.getLogger(n).setLevel(logging.ERROR)


# ═══════════════════════════════════════════════════════════
# 计时工具
# ═══════════════════════════════════════════════════════════
class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks = []
        self._last = self.t0

    def mark(self, name: str):
        now = time.perf_counter()
        self.marks.append((name, now - self._last, now - self.t0))
        self._last = now

    def report(self):
        print("\n" + "=" * 60)
        print("耗时报告 (分段 / 累计)")
        print("=" * 60)
        for name, seg, total in self.marks:
            print(f"  {name:<32} {seg:7.3f}s  |  累计 {total:7.3f}s")
        print("-" * 60)
        print(f"  {'TOTAL':<32} {self.marks[-1][2]:7.3f}s")
        print("=" * 60)


timer = Timer()

pdf = next((str(p) for p in Path(".").rglob("*.pdf") if "天梯" in p.name), None)
print(f"PDF: {pdf}")


# ═══════════════════════════════════════════════════════════
# 诊断三步（在 Docling 解析之前执行）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("诊断三步")
print("=" * 60)

# ── 步骤 1: 检查 PDF 是否有书签/目录 ──
import fitz  # PyMuPDF

fitz_doc = fitz.open(pdf)
toc = fitz_doc.get_toc()
print(f"\n[1] PDF 书签/目录数量: {len(toc)}")
if toc:
    for item in toc[:10]:
        print(f"    level={item[0]}  title={item[1]!r}  page={item[2]}")
    if len(toc) > 10:
        print(f"    ... 还有 {len(toc) - 10} 条")
else:
    print("    ⚠️ 无书签，heading_hierarchy_options 的 bookmark 信号将失效")

# ── 步骤 2: 检查 PDF 是否有文本层 ──
print(f"\n[2] PDF 文本层检查（前 3 页）")
for page_idx in range(min(3, len(fitz_doc))):
    page = fitz_doc[page_idx]
    text = page.get_text()
    print(f"    Page {page_idx + 1}: 字符数={len(text)}, 前 80 字={text[:80]!r}")
    if len(text.strip()) < 50:
        print(f"      ⚠️ 该页文本层极少，可能是扫描版")

fitz_doc.close()
timer.mark("诊断三步 (fitz)")


# ═══════════════════════════════════════════════════════════
# Docling 导入与解析
# ═══════════════════════════════════════════════════════════
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling_core.types.doc import SectionHeaderItem

timer.mark("import docling")

# ── 使用 ThreadedPdfPipelineOptions 获得批处理能力 ──
opts = ThreadedPdfPipelineOptions(
    do_ocr=True,
    ocr_batch_size=64,
    layout_batch_size=64,
    table_batch_size=4,
    generate_parsed_pages=True,
)
opts.accelerator_options = AcceleratorOptions(
    num_threads=8,
    device=AcceleratorDevice.AUTO,
)
opts.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
)
timer.mark("create converter")

doc = converter.convert(pdf).document
timer.mark("convert (模型+版面+OCR)")

# ── 构建 seq ──
seq = []
for item, level in doc.iterate_items():
    is_header = isinstance(item, SectionHeaderItem)
    text = (getattr(item, "text", "") or "").strip()
    sr = getattr(item, "self_ref", None)
    seq.append((item, level, is_header, text, sr))
timer.mark(f"iterate_items (seq={len(seq)})")

# ── header_has_content ──
header_has_content = {}
for i, (item, level, is_header, text, sr) in enumerate(seq):
    if not is_header or not text:
        continue
    has_content = False
    for j in range(i + 1, len(seq)):
        _, n_level, n_is_header, _, _ = seq[j]
        if not n_is_header:
            has_content = True
            break
        if n_level > level:
            continue
        break
    header_has_content[i] = has_content
timer.mark("header_has_content")

# ── chunking ──
chunker = HybridChunker()
chunks = list(chunker.chunk(dl_doc=doc))
timer.mark(f"chunking (chunks={len(chunks)})")


# ═══════════════════════════════════════════════════════════
# 步骤 3: 打印第一个 chunk 的完整 meta
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[3] 前 5 个 chunk 的 meta")
print("=" * 60)
for i, c in enumerate(chunks[:5]):
    print(f"\nChunk {i}:")
    print(f"  headings={c.meta.headings}")
    print(f"  doc_items refs={[getattr(it, 'self_ref', None) for it in (c.meta.doc_items or [])]}")
    print(f"  text={c.text[:80]!r}")


# ═══════════════════════════════════════════════════════════
# 原有回溯逻辑
# ═══════════════════════════════════════════════════════════
ref_to_seq_idx = {sr: i for i, (_, _, _, _, sr) in enumerate(seq) if sr}

fixed = 0
for c in chunks:
    items = getattr(c.meta, "doc_items", None) or []
    if not items:
        continue
    first_ref = getattr(items[0], "self_ref", None)
    first_idx = ref_to_seq_idx.get(first_ref)
    if first_idx is None:
        continue
    extra = []
    for i in range(first_idx - 1, -1, -1):
        item, level, is_header, text, sr = seq[i]
        if not is_header or not text:
            continue
        if header_has_content.get(i, True):
            break
        extra.insert(0, text)
    if not extra:
        continue
    existing = list(getattr(c.meta, "headings", None) or [])
    new_headings = [h for h in extra if h not in existing] + existing
    if new_headings != existing:
        c.meta.headings = new_headings
        fixed += 1
timer.mark(f"backtrack 补全 (fixed={fixed})")


# ═══════════════════════════════════════════════════════════
# 第 24 页附近的 seq
# ═══════════════════════════════════════════════════════════
print("\n=== 第 24 页附近的 seq ===")
for i, (item, level, is_header, text, sr) in enumerate(seq):
    if sr and ('#/texts/654' == sr or '#/texts/650' <= sr <= '#/texts/670'):
        print(f"  idx={i} sr={sr!r} is_header={is_header} level={level} "
              f"has_content={header_has_content.get(i, 'N/A')} text={text[:40]!r}")


# ═══════════════════════════════════════════════════════════
# 第 24 页 chunk 回溯
# ═══════════════════════════════════════════════════════════
print("\n=== 第 24 页的 chunk 回溯 ===")
for c in chunks:
    items = getattr(c.meta, "doc_items", None) or []
    if not items:
        continue
    pages = []
    for it in items:
        for p in (getattr(it, "prov", None) or []):
            pn = getattr(p, "page_no", None)
            if pn is not None:
                pages.append(pn)
    if 24 not in pages:
        continue
    first = items[0]
    first_ref = getattr(first, "self_ref", None)
    first_idx = ref_to_seq_idx.get(first_ref)
    print(f"\nchunk first_ref={first_ref!r} first_idx={first_idx}")
    if first_idx is None:
        print("  ❌ first_ref 不在 ref_to_seq_idx")
        continue
    extra = []
    for i in range(first_idx - 1, -1, -1):
        item, level, is_header, text, sr = seq[i]
        if not is_header or not text:
            continue
        hc = header_has_content.get(i, True)
        print(f"  i={i} sr={sr!r} is_header={is_header} has_content={hc} text={text[:40]!r}")
        if hc:
            print("  → 停止（有内容标题）")
            break
        extra.insert(0, text)
    print(f"  extra_ancestors={extra}")


timer.report()