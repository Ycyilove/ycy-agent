"""多文档解析导出：下载网络 PDF + 本地 PDF，按 dump_chunks 的格式落 JSONL。

每份 PDF 输出一个 <stem>.chunks.jsonl 到 data/compare_pdfs/，
字段与 scripts/dump_chunks.py 对齐，便于逐条对比 heading 和文本。

用法：
    python scripts/compare_parsers.py
    python scripts/compare_parsers.py --local "第十届….pdf"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.pdf_fast_parser import parse_fast, extract_tables

CACHE_DIR = _ROOT / "data" / "compare_pdfs"

REMOTE_PDFS = {
    "berkshire_2023ar.pdf": "https://www.berkshirehathaway.com/2023ar/2023ar.pdf",
    "rag_2005.11401.pdf": "https://arxiv.org/pdf/2005.11401",
}


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] 已缓存: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dl] {dest.name} <- {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def export_chunks_jsonl(pdf_path: Path, out_dir: Path) -> dict:
    """跑 parse_fast，按 dump_chunks 的格式落 JSONL。返回汇总指标。"""
    chunks = parse_fast(str(pdf_path))
    tables = extract_tables(str(pdf_path))

    out_path = out_dir / f"{pdf_path.stem}.chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            record = {
                "idx": i,
                "headings": list(c.headings or []),
                "page_numbers": [c.page_no] if c.page_no is not None else [],
                "original_text": c.text,
                "chunk_chars": len(c.text),
                "font_size": c.font_size,
                "is_bold": c.is_bold,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = len(chunks)
    no_heading = sum(1 for c in chunks if not c.headings)
    chars = [len(c.text) for c in chunks]
    all_headings = [h for c in chunks for h in (c.headings or [])]
    uniq_headings = set(all_headings)
    avg_heading_len = (
        sum(len(h) for h in uniq_headings) / len(uniq_headings)
        if uniq_headings else 0
    )

    return {
        "file": pdf_path.name,
        "chunks": n,
        "no_heading": no_heading,
        "uniq_headings": len(uniq_headings),
        "avg_heading_len": round(avg_heading_len),
        "min_chars": min(chars) if chars else 0,
        "avg_chars": round(sum(chars) / n) if n else 0,
        "max_chars": max(chars) if chars else 0,
        "tables": len(tables),
        "out": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="第十届“中国高校计算机大赛-团体程序设计天梯赛”全国总决赛获奖名单.pdf")
    args = ap.parse_args()

    out_dir = CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = []
    for name, url in REMOTE_PDFS.items():
        try:
            pdf_paths.append(download(url, CACHE_DIR / name))
        except Exception as e:
            print(f"[✗] 下载失败 {name}: {e}")

    local = (_ROOT / args.local).resolve()
    if local.exists():
        pdf_paths.append(local)
    else:
        print(f"[✗] 本地文件不存在: {local}")

    rows = []
    for p in pdf_paths:
        print(f"[...] 解析并导出 {p.name}")
        try:
            rows.append(export_chunks_jsonl(p, out_dir))
        except Exception as e:
            print(f"[✗] {p.name} 失败: {e}")

    if not rows:
        return

    headers = ["file", "chunks", "no_heading", "uniq_headings", "avg_heading_len",
               "min_chars", "avg_chars", "max_chars", "tables"]
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(widths[h]) for h in headers))

    print("\n导出文件：")
    for r in rows:
        print(f"  {r['out']}")


if __name__ == "__main__":
    main()