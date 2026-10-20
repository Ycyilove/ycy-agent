"""把指定文档的 chunk 和 Contextual 摘要 dump 到 JSONL。

自动分流：
    - 有文本层的 PDF → PyMuPDF 快路径（秒级）
    - 无文本层的 PDF → Docling + OCR

用法：
    python -m tools.dump_chunks "path/to/xxx.pdf"
    python -m tools.dump_chunks "path/to/xxx.pdf" --no-ctx
    python -m tools.dump_chunks "path/to/xxx.pdf" --out out/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "models"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"])
os.environ["HF_HUB_DISABLE_XET"] = "1"

for n in ["docling", "httpx", "httpcore", "transformers", "RapidOCR",
          "jieba", "openai", "langchain_openai", "asyncio"]:
    logging.getLogger(n).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dump_chunks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="PDF 文件路径")
    parser.add_argument("--out", default=None, help="输出 JSONL 路径")
    parser.add_argument("--no-ctx", action="store_true", help="跳过 Contextual 摘要")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"[✗] PDF 不存在: {pdf_path}")
        sys.exit(1)

    out_path = (
        Path(args.out).resolve()
        if args.out
        else pdf_path.with_suffix(".chunks.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. 解析（自动分流）──
    from rag.pdf_fast_parser import has_text_layer
    _has_text = has_text_layer(str(pdf_path))
    _route = "PyMuPDF 快路径" if _has_text else "Docling + OCR"
    print(f"[1] 解析: {pdf_path.name}  (路径: {_route})")

    t0 = time.perf_counter()
    from rag import DoclingParser
    parser = DoclingParser()
    docling_chunks = parser.parse_to_chunks(str(pdf_path))
    elapsed = time.perf_counter() - t0
    print(f"    得到 {len(docling_chunks)} 个 chunk  (耗时 {elapsed:.1f}s)")

    # ── 2. 提取原文 + headings ──
    from rag.text_utils import clean_cn_spaces, clean_heading

    originals = []
    headings_list = []
    pages_list = []
    for dc in docling_chunks:
        originals.append(clean_cn_spaces(dc.text or ""))
        h_list = [clean_heading(h) for h in (getattr(dc.meta, "headings", None) or [])]
        headings_list.append(h_list)
        pages_list.append(list(getattr(dc.meta, "page_numbers", None) or []))

    # ── 3. Contextual 摘要（可选）──
    summaries = [""] * len(docling_chunks)
    enriched_list = [None] * len(docling_chunks)

    if not args.no_ctx:
        print(f"[2] Contextual 摘要生成")
        try:
            from backend.config import (
                RAG_CONTEXTUAL_CONCURRENCY,
                RAG_CONTEXTUAL_TIMEOUT,
                RAG_CONTEXTUAL_MIN_CHARS,
            )
            from rag.contextualizer import ChunkContextualizer

            ctx_inputs = [
                {"text": originals[i], "headings": headings_list[i]}
                for i in range(len(originals))
            ]
            ctx = ChunkContextualizer(
                max_workers=RAG_CONTEXTUAL_CONCURRENCY,
                timeout=RAG_CONTEXTUAL_TIMEOUT,
                min_chunk_chars=RAG_CONTEXTUAL_MIN_CHARS,
            )
            ctx_results = ctx.contextualize_batch(ctx_inputs)
            for i, r in enumerate(ctx_results):
                summaries[i] = r.get("summary", "")
                enriched_list[i] = r.get("enriched_text", "")
            n_ok = sum(1 for s in summaries if s)
            print(f"    {n_ok}/{len(summaries)} 生成摘要")
        except Exception:
            logger.exception("Contextual 摘要生成失败，跳过")
    else:
        print("[2] 跳过 Contextual 摘要（--no-ctx）")

    # ── 4. 落盘 ──
    print(f"[3] 写入: {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        for i, dc in enumerate(docling_chunks):
            if enriched_list[i]:
                enriched = enriched_list[i]
            else:
                enriched = parser.get_enriched_text(dc)

            record = {
                "idx": i,
                "headings": headings_list[i],
                "heading_levels": list(
                    getattr(dc.meta, "heading_levels", None) or []
                ),
                "page_numbers": pages_list[i],
                "original_text": originals[i],
                "enriched_text": enriched,
                "summary": summaries[i],
                "chunk_chars": len(originals[i]),
                "summary_chars": len(summaries[i]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[✓] 完成，共 {len(docling_chunks)} 条")
    print(f"    输出: {out_path}")


if __name__ == "__main__":
    main()