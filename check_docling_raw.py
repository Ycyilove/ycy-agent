# -*- coding: utf-8 -*-
"""一次性排查：Docling 原始输出里的"团队" + PdfPipelineOptions / layout_options 字段。

用法：
    python check_docling_raw.py
    python check_docling_raw.py path/to/xxx.pdf
"""

import os
import sys
import logging
from pathlib import Path

# ── 强制 stdout UTF-8，避免 GBK 崩溃 ──
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 走镜像，但【不强制离线】——Docling 需要确认模型版本 ──
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── 压掉第三方日志刷屏 ──
for name in [
    "docling", "docling.datamodel", "docling.document_converter",
    "docling.models", "docling.pipeline", "docling.utils",
    "docling.models.factories", "docling.models.inference_engines",
    "docling.models.stages", "docling.models.utils",
    "httpx", "httpcore", "urllib3",
    "transformers", "sentence_transformers", "faiss",
    "jieba", "RapidOCR", "onnxruntime",
    "huggingface_hub", "filelock",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def find_pdf() -> str | None:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        return str(p) if p.is_file() else None
    for p in Path(".").rglob("*.pdf"):
        if "天梯" in p.name or "获奖" in p.name:
            return str(p)
    return None


def sep(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# 1. PdfPipelineOptions 的所有字段
# ─────────────────────────────────────────────────────────────

def dump_pipeline_options():
    sep("[1] PdfPipelineOptions 所有字段")

    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError as e:
        print(f"❌ import 失败: {e}")
        return

    try:
        fields = PdfPipelineOptions.model_fields
    except AttributeError:
        fields = getattr(PdfPipelineOptions, "__fields__", {})

    if not fields:
        print("⚠️ 拿不到字段，尝试实例化后看属性")
        try:
            opts = PdfPipelineOptions()
            for k in dir(opts):
                if not k.startswith("_"):
                    v = getattr(opts, k)
                    if not callable(v):
                        print(f"  {k} = {v!r}")
        except Exception as e:
            print(f"❌ {e}")
        return

    for name, field in fields.items():
        ann = getattr(field, "annotation", "?")
        default = getattr(field, "default", "?")
        d = repr(default)
        if len(d) > 80:
            d = d[:77] + "..."
        print(f"  {name}: {ann} = {d}")

    # ── 页眉/页脚相关字段 ──
    print()
    print("  ── 页眉/页脚相关字段 ──")
    hit = False
    for name in fields:
        low = name.lower()
        if any(k in low for k in ["header", "footer", "page", "ocr", "layout"]):
            print(f"    ⭐ {name}")
            hit = True
    if not hit:
        print("    (无)")


# ─────────────────────────────────────────────────────────────
# 2. layout_options 的字段和默认实例
# ─────────────────────────────────────────────────────────────

def dump_layout_options():
    sep("[2] layout_options 的字段和默认实例")

    try:
        from docling.datamodel.pipeline_options import BaseLayoutOptions
    except ImportError as e:
        print(f"❌ import BaseLayoutOptions 失败: {e}")
        return

    # ── 字段 ──
    print("── BaseLayoutOptions 字段 ──")
    try:
        fields = BaseLayoutOptions.model_fields
    except AttributeError:
        fields = getattr(BaseLayoutOptions, "__fields__", {})

    if fields:
        for name, field in fields.items():
            ann = getattr(field, "annotation", "?")
            default = getattr(field, "default", "?")
            d = repr(default)
            if len(d) > 80:
                d = d[:77] + "..."
            print(f"  {name}: {ann} = {d}")
    else:
        print("  (拿不到字段)")

    # ── 默认实例 ──
    print()
    print("── 默认 layout_options 实例 ──")
    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        opts = PdfPipelineOptions()
        lo = opts.layout_options
        print(f"  type: {type(lo).__name__}")
        if hasattr(lo, "__dict__"):
            for k, v in lo.__dict__.items():
                print(f"    {k} = {v!r}")
        else:
            for k in dir(lo):
                if not k.startswith("_"):
                    v = getattr(lo, k)
                    if not callable(v):
                        print(f"    {k} = {v!r}")
    except Exception as e:
        print(f"  ❌ {e}")


# ─────────────────────────────────────────────────────────────
# 3. Docling 原始输出里"团队"/"全国"是什么类型
# ─────────────────────────────────────────────────────────────

def dump_docling_raw(pdf_path: str):
    sep("[3] Docling 原始输出里的 '团队' / '全国'")

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        HeadingHierarchyOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # ── 用和 rag/docling_parser.py 一致的配置 ──
    pipeline_options = PdfPipelineOptions()
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(
        enabled=True,
    )
    pipeline_options.generate_parsed_pages = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )

    print(f"解析中: {pdf_path}（可能要 1~3 分钟，OCR 慢）")
    result = converter.convert(pdf_path)
    doc = result.document
    print("解析完成")

    # ── 含"团队"的元素 ──
    print()
    print("── 含 '团队' 的元素 ──")
    hits_team = 0
    for item, level in doc.iterate_items():
        text = getattr(item, "text", "") or ""
        if "团队" in text:
            hits_team += 1
            cls = type(item).__name__
            print(f"  type={cls} level={level}")
            print(f"    text={text!r}")
            prov = getattr(item, "prov", None)
            if prov:
                for p in prov:
                    page_no = getattr(p, "page_no", None)
                    bbox = getattr(p, "bbox", None)
                    print(f"    page_no={page_no} bbox={bbox}")
            print()
    if hits_team == 0:
        print("  (无)")

    # ── 含"全国"的元素（短行）──
    print()
    print("── 含 '全国' 的元素（len<40）──")
    hits_nation = 0
    for item, level in doc.iterate_items():
        text = getattr(item, "text", "") or ""
        if "全国" in text and len(text) < 40:
            hits_nation += 1
            cls = type(item).__name__
            print(f"  type={cls} level={level}")
            print(f"    text={text!r}")
            prov = getattr(item, "prov", None)
            if prov:
                for p in prov:
                    page_no = getattr(p, "page_no", None)
                    print(f"    page_no={page_no}")
            if hits_nation >= 30:
                print("  ... (只显示前 30)")
                break
    if hits_nation == 0:
        print("  (无)")

    # ── 第 24 页的所有元素 ──
    print()
    print("── 第 24 页的所有元素 ──")
    hits_p24 = 0
    for item, level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        pages = [getattr(p, "page_no", None) for p in prov]
        if 24 not in pages:
            continue
        hits_p24 += 1
        cls = type(item).__name__
        text = getattr(item, "text", "") or ""
        text_short = text[:80].replace("\n", " ")
        print(f"  type={cls} level={level} pages={pages}")
        print(f"    text={text_short!r}")
        if hits_p24 >= 30:
            print("  ... (只显示前 30)")
            break
    if hits_p24 == 0:
        print("  (无)")

    # ── 统计所有 item 类型 ──
    print()
    print("── 所有 item 类型统计 ──")
    from collections import Counter
    counter = Counter()
    for item, _ in doc.iterate_items():
        counter[type(item).__name__] += 1
    for cls, n in counter.most_common():
        print(f"  {cls}: {n}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    pdf_path = find_pdf()

    dump_pipeline_options()
    dump_layout_options()

    if pdf_path is None:
        print()
        print("⚠️ 找不到 PDF，跳过 [3]")
        print("   用法: python check_docling_raw.py path/to/xxx.pdf")
        return

    print()
    print(f"PDF: {pdf_path}")
    dump_docling_raw(pdf_path)

    sep("完成")
    print("把 [1] [2] [3] 的输出发给我")
    print("重点：[2] BaseLayoutOptions 字段 + [3] '团队' 的 item 类型")


if __name__ == "__main__":
    main()