"""对比 EasyOCR 与 PP-OCRv6 对层级解析的影响。

用法：
    python test_ocr_engines.py "path/to/xxx.pdf"

依赖：
    pip install easyocr
    pip install "docling-pp-ocrv6[cpu]"
"""

import os
import sys
import time
import logging
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "models"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"])
os.environ["HF_HUB_DISABLE_XET"] = "1"

for n in ["docling", "httpx", "httpcore", "transformers", "RapidOCR", "easyocr"]:
    logging.getLogger(n).setLevel(logging.WARNING)


def build_pipeline_options(engine_name: str):
    """构建指定引擎的 pipeline options。"""
    from docling.datamodel.pipeline_options import (
        ThreadedPdfPipelineOptions,
        HeadingHierarchyOptions,
    )
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )

    opts = ThreadedPdfPipelineOptions(
        do_ocr=True,
        ocr_batch_size=64,
        layout_batch_size=64,
        table_batch_size=4,
        generate_parsed_pages=True,
    )

    if engine_name == "easyocr":
        from docling.datamodel.pipeline_options import EasyOcrOptions
        opts.ocr_options = EasyOcrOptions(
            lang=["ch_sim", "en"],
            use_gpu=False,
            force_full_page_ocr=True,
        )
    elif engine_name == "ppocrv6":
        from docling_pp_ocrv6 import PPOCRv6Options
        opts.ocr_options = PPOCRv6Options(
            lang=["ch", "en"],
        )
    else:
        raise ValueError(f"未知引擎: {engine_name}")

    opts.heading_hierarchy_options = HeadingHierarchyOptions(
        enabled=True,
        use_bookmarks=True,
        use_numbering=True,
        use_style=True,
        bookmark_match_threshold=0.8,
    )
    opts.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.AUTO,
    )
    return opts


def analyze_headings(doc) -> dict:
    """分析文档中的 heading 错误父子关系。

    判定规则：相邻两个 heading 都是"学校名"，
    且后者 level 大于前者，视为错误父子关系。
    """
    from docling_core.types.doc import SectionHeaderItem

    SCHOOL_KEYWORDS = [
        "大学", "学院", "职业技术学院",
        "（珠峰争鼎）", "（华山论剑）", "（沧海竞舟）",
    ]

    def is_school_like(text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        for kw in SCHOOL_KEYWORDS:
            if kw in text:
                return True
        return text.endswith("大学") or text.endswith("学院")

    headings_in_order = []
    for item, level in doc.iterate_items():
        if isinstance(item, SectionHeaderItem):
            text = (getattr(item, "text", "") or "").strip()
            if text:
                headings_in_order.append((text, level))

    errors = []
    for i in range(len(headings_in_order) - 1):
        h1, l1 = headings_in_order[i]
        h2, l2 = headings_in_order[i + 1]
        if is_school_like(h1) and is_school_like(h2) and l2 > l1:
            errors.append((h1, h2, l1, l2))

    return {
        "total_headings": len(headings_in_order),
        "school_like": sum(1 for h, _ in headings_in_order if is_school_like(h)),
        "errors": errors,
        "error_count": len(errors),
    }


def run_engine(engine_name: str, pdf_path: str):
    """用指定引擎解析 PDF 并输出对比指标。"""
    print(f"\n{'=' * 70}")
    print(f"引擎: {engine_name}")
    print(f"{'=' * 70}")

    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    try:
        opts = build_pipeline_options(engine_name)
    except ImportError as e:
        print(f"  [跳过] 缺少依赖: {e}")
        return None

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    t0 = time.perf_counter()
    try:
        result = converter.convert(pdf_path)
    except Exception as e:
        print(f"  [失败] 转换异常: {e}")
        return None
    elapsed = time.perf_counter() - t0

    doc = result.document

    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        max_tokens=256,
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    chunks = list(chunker.chunk(dl_doc=doc))

    stats = analyze_headings(doc)

    print(f"  解析耗时: {elapsed:.1f}s")
    print(f"  chunk 数量: {len(chunks)}")
    print(f"  heading 总数: {stats['total_headings']}")
    print(f"  学校类 heading: {stats['school_like']}")
    print(f"  ⚠️ 疑似错误父子对: {stats['error_count']}")
    if stats["errors"]:
        print(f"  前 5 个错误示例:")
        for h1, h2, l1, l2 in stats["errors"][:5]:
            print(f"    [{l1}→{l2}] {h1[:30]!r}  →  {h2[:30]!r}")

    # dump 到 jsonl 供后续对比
    out_path = Path(pdf_path).with_suffix(f".{engine_name}.chunks.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        import json
        for i, c in enumerate(chunks):
            headings = list(getattr(c.meta, "headings", None) or [])
            f.write(json.dumps({
                "idx": i,
                "headings": headings,
                "text": c.text[:200],
            }, ensure_ascii=False) + "\n")
    print(f"  chunk 已 dump: {out_path}")

    return {
        "engine": engine_name,
        "elapsed": elapsed,
        "chunk_count": len(chunks),
        "total_headings": stats["total_headings"],
        "school_like": stats["school_like"],
        "error_count": stats["error_count"],
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python test_ocr_engines.py <pdf路径>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"PDF 不存在: {pdf_path}")
        sys.exit(1)

    engines = ["easyocr", "ppocrv6"]
    results = []

    for engine in engines:
        r = run_engine(engine, pdf_path)
        if r:
            results.append(r)

    print(f"\n{'=' * 70}")
    print("汇总对比")
    print(f"{'=' * 70}")
    print(f"{'引擎':<12} {'耗时':>10} {'chunk数':>9} {'heading数':>11} {'学校类':>8} {'错误父子对':>12}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['engine']:<12} {r['elapsed']:>9.1f}s {r['chunk_count']:>9} "
            f"{r['total_headings']:>11} {r['school_like']:>8} {r['error_count']:>12}"
        )

    if results:
        best = min(results, key=lambda x: x["error_count"])
        print(f"\n✅ 错误父子对最少: {best['engine']} ({best['error_count']} 个)")


if __name__ == "__main__":
    main()