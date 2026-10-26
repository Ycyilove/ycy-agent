"""全文件切分质量测试 + 导出。

用法：
    # 快速版：只测 PyMuPDF（约 3 秒）
    python -m tests.test_parse_quality "xxx.pdf" --fast-only

    # 完整版：PyMuPDF + Docling 对比（Docling 可能 60s+）
    python -m tests.test_parse_quality "xxx.pdf"

输出：
    <pdf_stem>.pymupdf_plain.chunks.jsonl
    <pdf_stem>.pymupdf_heading.chunks.jsonl
    <pdf_stem>.docling.chunks.jsonl
    <pdf_stem>.quality_report.json      # 全量质量报告
    <pdf_stem>.chunk_distribution.txt   # 分页 chunk 分布（人工审阅用）
"""

from __future__ import annotations

import sys
import json
import time
import logging
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_parse_quality")


# ═════════════════════════════════════════════════════════════
# 质量分析：全文件指标
# ═════════════════════════════════════════════════════════════

def analyze_chunks(chunks, pdf_path: str) -> dict:
    import fitz

    texts = [(getattr(c, "text", "") or "").strip() for c in chunks]
    texts = [t for t in texts if t]
    n = len(texts)
    if n == 0:
        return {"error": "no_chunks"}

    lens = sorted(len(t) for t in texts)
    def pct(p):
        idx = min(int(len(lens) * p), len(lens) - 1)
        return lens[idx]
    length_stats = {
        "min": lens[0], "p10": pct(0.10), "p25": pct(0.25), "p50": pct(0.50),
        "p75": pct(0.75), "p90": pct(0.90), "max": lens[-1],
        "mean": sum(lens) / n,
    }
    short_1 = sum(1 for t in texts if len(t) == 1)
    short_2 = sum(1 for t in texts if len(t) == 2)
    short_3_10 = sum(1 for t in texts if 3 <= len(t) <= 10)
    normal = sum(1 for t in texts if len(t) > 10)
    short_stats = {
        "len=1": short_1, "len=2": short_2, "len=3-10": short_3_10, "len>10": normal,
        "short_ratio(<=2)": (short_1 + short_2) / n,
        "short_ratio(<=10)": (short_1 + short_2 + short_3_10) / n,
    }

    doc = fitz.open(pdf_path)
    raw = ""
    page_raw_chars = {}
    for pno, page in enumerate(doc, start=1):
        t = page.get_text()
        raw += t
        page_raw_chars[pno] = set(t.replace(" ", "").replace("\n", ""))
    doc.close()

    raw_set = set(raw.replace(" ", "").replace("\n", ""))
    chunk_set = set("".join(texts).replace(" ", "").replace("\n", ""))
    coverage = len(raw_set & chunk_set) / max(1, len(raw_set))
    missing_chars = raw_set - chunk_set

    # ── P0：chunk 覆盖的页集合（三级回退）──
    chunk_pages = set()
    for c in chunks:
        meta = getattr(c, "meta", None)
        pages = []

        # 1. 快路径：_wrap_fast_chunks 包装的 _Meta 有 page_numbers
        if meta is not None:
            pages = list(getattr(meta, "page_numbers", None) or [])

        # 2. Docling 路径：DocMeta 无 page_numbers，从 doc_items[].prov[].page_no 提取
        if not pages and meta is not None:
            for item in (getattr(meta, "doc_items", None) or []):
                for p in (getattr(item, "prov", None) or []):
                    pno = getattr(p, "page_no", None)
                    if pno is not None:
                        pages.append(pno)

        # 3. 兜底：FastChunk 的 page_no
        if not pages:
            pno = getattr(c, "page_no", None)
            if pno is not None:
                pages = [pno]

        chunk_pages.update(p for p in pages if p is not None)

    chunk_text_all = "".join(texts)
    page_coverage = {}
    no_chunk_pages = []
    for pno, chars in page_raw_chars.items():
        if not chars:
            page_coverage[pno] = "empty_page"
            continue
        if pno not in chunk_pages:
            # ── P0：该页有文本，但没有任何 chunk 声称覆盖它 ──
            no_chunk_pages.append(pno)
            page_coverage[pno] = "no_chunk"
            continue
        found = sum(1 for ch in chars if ch in chunk_text_all)
        ratio = found / len(chars)
        if ratio >= 0.95:
            page_coverage[pno] = "full"
        elif ratio >= 0.5:
            page_coverage[pno] = f"partial({ratio:.0%})"
        else:
            page_coverage[pno] = f"missing({ratio:.0%})"

    coherent = sum(1 for t in texts if any(p in t for p in "。！？、，；\n"))
    coherence = coherent / n

    # ── 6. 综合质量分 ──
    frag_penalty = short_stats["short_ratio(<=2)"]
    avg_len = length_stats["mean"]
    p50 = length_stats["p50"]

    # ── P2：len_score 同时看 mean 和 p50，防止"均匀碎片化"骗分 ──
    # 旧逻辑只看 mean：docling p50=50、mean=140 仍得满分 1.0，
    # 掩盖了"一半 chunk 只有 50 字"的碎片化。
    # 新逻辑：p50 < 100 时按 p50 比例扣分（最低 0.3），
    # 再叠加原 mean 逻辑取较小值，双重约束。
    if avg_len < 10:
        len_score_mean = avg_len / 10
    elif avg_len <= 500:
        len_score_mean = 1.0
    else:
        len_score_mean = max(0.3, 500 / avg_len)

    if p50 < 100:
        # 中位 chunk 不足 100 字 → 均匀碎片化，按比例惩罚
        len_score_p50 = max(0.3, p50 / 100.0)
    elif p50 <= 500:
        len_score_p50 = 1.0
    else:
        len_score_p50 = max(0.3, 500.0 / p50)

    len_score = min(len_score_mean, len_score_p50)

    score = (
        0.40 * (1 - frag_penalty)
        + 0.30 * coverage
        + 0.20 * coherence
        + 0.10 * len_score
    )

    return {
        "n_chunks": n,
        "length_stats": length_stats,
        "short_stats": short_stats,
        "coverage": coverage,
        "coherence": coherence,
        "score": score,
        "missing_chars_count": len(missing_chars),
        "missing_chars_sample": "".join(sorted(missing_chars)[:50]),
        "page_coverage_summary": dict(Counter(page_coverage.values())),
        "page_coverage_detail": page_coverage,
        "no_chunk_pages": no_chunk_pages,   # ── P0 新增 ──
    }

# ═════════════════════════════════════════════════════════════
# 导出 JSONL
# ═════════════════════════════════════════════════════════════

def export_jsonl(chunks, out_path: Path):
    """导出 chunks 到 JSONL（与生产 dump_chunks 格式一致）。"""
    with out_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            text = (getattr(c, "text", "") or "").strip()
            if not text:
                continue
            meta = getattr(c, "meta", None)
            headings = list(getattr(meta, "headings", None) or []) if meta else \
                       list(getattr(c, "headings", None) or [])
            page_numbers = list(getattr(meta, "page_numbers", None) or []) if meta else \
                           ([c.page_no] if getattr(c, "page_no", None) else [])

            record = {
                "idx": i,
                "headings": headings,
                "page_numbers": page_numbers,
                "original_text": text,
                "chunk_chars": len(text),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("[export] %d chunks → %s", len(chunks), out_path)


# ═════════════════════════════════════════════════════════════
# 分页分布报告（人工审阅用）
# ═════════════════════════════════════════════════════════════

def export_distribution(chunks, report: dict, out_path: Path):
    by_page = defaultdict(list)
    for c in chunks:
        meta = getattr(c, "meta", None)
        pages = list(getattr(meta, "page_numbers", None) or []) if meta else \
                ([c.page_no] if getattr(c, "page_no", None) else [])
        text = (getattr(c, "text", "") or "").strip()
        for p in pages:
            by_page[p].append(text)

    # ── P0：把 PDF 中存在但 by_page 无 chunk 的页补进来 ──
    all_pdf_pages = set(report["page_coverage_detail"].keys())
    for p in sorted(all_pdf_pages - set(by_page.keys())):
        by_page[p] = []   # 空列表，下面会写 no_chunk

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Chunk 分布报告\n\n")
        f.write(f"## 总体指标\n\n")
        f.write(f"- chunk 总数: {report['n_chunks']}\n")
        f.write(f"- 质量分: {report['score']:.3f}\n")
        f.write(f"- 覆盖率: {report['coverage']:.2%}\n")
        f.write(f"- 连贯性: {report['coherence']:.2%}\n")
        f.write(f"- 短 chunk 占比(<=2): {report['short_stats']['short_ratio(<=2)']:.2%}\n")
        f.write(f"- 长度分布: min={report['length_stats']['min']}, "
                f"p25={report['length_stats']['p25']}, "
                f"p50={report['length_stats']['p50']}, "
                f"p90={report['length_stats']['p90']}, "
                f"max={report['length_stats']['max']}\n")
        ncp = report.get("no_chunk_pages", [])
        if ncp:
            f.write(f"- ⚠️ 无 chunk 覆盖页: {ncp}\n")
        f.write("\n## 分页覆盖概况\n\n")
        for status, cnt in sorted(report['page_coverage_summary'].items()):
            f.write(f"- {status}: {cnt} 页\n")
        f.write("\n## 逐页 chunk 详情\n\n")
        for pno in sorted(by_page):
            texts = by_page[pno]
            status = report['page_coverage_detail'].get(pno, "unknown")
            f.write(f"### Page {pno} [{status}] — {len(texts)} chunks\n\n")
            if not texts:
                f.write(f"  ⚠️ 该页有文本但无任何 chunk 覆盖（检查跨页归属）\n\n")
                continue
            for i, t in enumerate(texts):
                preview = t.replace("\n", "\\n")[:100]
                f.write(f"  [{i}] ({len(t)}字) {preview}\n")
            f.write("\n")
        missing_pages = [p for p, s in report['page_coverage_detail'].items()
                         if isinstance(s, str) and s.startswith("missing")]
        if missing_pages:
            f.write(f"## ⚠️ 缺失页清单\n\n{missing_pages}\n")
        partial_pages = [p for p, s in report['page_coverage_detail'].items()
                         if isinstance(s, str) and s.startswith("partial")]
        if partial_pages:
            f.write(f"\n## ⚠️ 部分缺失页清单\n\n{partial_pages}\n")
    logger.info("[export] 分布报告 → %s", out_path)

# ═════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("用法: python -m tests.test_parse_quality <pdf路径> [--fast-only]")
        sys.exit(1)

    pdf_path = str(Path(sys.argv[1]).resolve())
    if not Path(pdf_path).exists():
        print(f"[✗] PDF 不存在: {pdf_path}")
        sys.exit(1)

    fast_only = "--fast-only" in sys.argv
    # ── P3：默认 False，显式传 --skip-docling-if-plain-ok 才短路 ──
    skip_docling_if_plain_ok = "--skip-docling-if-plain-ok" in sys.argv
    pdf_stem = Path(pdf_path).stem
    out_dir = Path(pdf_path).parent

    print(f"\n{'='*70}")
    print(f"测试文件: {pdf_path}")
    print(f"输出目录: {out_dir}")
    print(f"模式: {'快速版(只 PyMuPDF)' if fast_only else '完整版(含 Docling)'}")
    print(f"Docling 短路: {'开启' if skip_docling_if_plain_ok else '关闭（默认跑 Docling）'}")
    print(f"{'='*70}\n")

    all_results = {}

    # ── 候选 A：PyMuPDF 纯文本 ──
    print("[1/3] PyMuPDF parse_fast(detect_headings=False) ...")
    plain_score = 0.0
    try:
        from rag.pdf_fast_parser import parse_fast
        from rag.docling_parser import DoclingParser
        t0 = time.perf_counter()
        fast_plain = parse_fast(pdf_path, detect_headings=False)
        t_cost = time.perf_counter() - t0
        wrapped = DoclingParser()._wrap_fast_chunks(fast_plain)
        report = analyze_chunks(wrapped, pdf_path)
        report["time"] = t_cost
        export_jsonl(wrapped, out_dir / f"{pdf_stem}.pymupdf_plain.chunks.jsonl")
        all_results["pymupdf_plain"] = {"chunks": wrapped, "report": report}
        plain_score = report["score"]
        print(f"   score={report['score']:.3f}, 耗时={t_cost:.1f}s, "
              f"chunks={report['n_chunks']}, "
              f"短chunk={report['short_stats']['short_ratio(<=2)']:.1%}, "
              f"覆盖率={report['coverage']:.1%}, "
              f"无chunk页={len(report.get('no_chunk_pages', []))}")
    except Exception as e:
        logger.exception("PyMuPDF 纯文本失败")
        all_results["pymupdf_plain"] = {"error": str(e)}

    # ── 候选 B：PyMuPDF + heading（P1：plain 达标则跳过）──
    if plain_score >= 0.75:
        print(f"\n[2/3] PyMuPDF heading 跳过（plain score={plain_score:.3f} >= 0.75，"
              f"与生产短路策略一致）")
        all_results["pymupdf_heading"] = {"skipped": True, "reason": "plain_score>=0.75"}
    else:
        print("\n[2/3] PyMuPDF parse_fast(detect_headings=True) ...")
        try:
            t0 = time.perf_counter()
            fast_heading = parse_fast(pdf_path, detect_headings=True)
            t_cost = time.perf_counter() - t0
            wrapped_h = DoclingParser()._wrap_fast_chunks(fast_heading)
            report_h = analyze_chunks(wrapped_h, pdf_path)
            report_h["time"] = t_cost
            export_jsonl(wrapped_h, out_dir / f"{pdf_stem}.pymupdf_heading.chunks.jsonl")
            all_results["pymupdf_heading"] = {"chunks": wrapped_h, "report": report_h}
            print(f"   score={report_h['score']:.3f}, 耗时={t_cost:.1f}s, "
                  f"chunks={report_h['n_chunks']}, "
                  f"短chunk={report_h['short_stats']['short_ratio(<=2)']:.1%}, "
                  f"覆盖率={report_h['coverage']:.1%}, "
                  f"无chunk页={len(report_h.get('no_chunk_pages', []))}")
        except Exception as e:
            logger.exception("PyMuPDF heading 失败")
            all_results["pymupdf_heading"] = {"error": str(e)}

    # ── 候选 C：Docling ──
    # ── P3：显式开关，默认不短路（False），方便验证 P0/P1 效果 ──
    plain_score = (
        all_results.get("pymupdf_plain", {}).get("report", {}).get("score", 0.0)
    )
    if fast_only:
        print("\n[3/3] Docling 跳过（--fast-only）")
        all_results["docling"] = {"skipped": True}
    elif skip_docling_if_plain_ok and plain_score >= 0.75:
        print(
            f"\n[3/3] Docling 跳过（--skip-docling-if-plain-ok 且 "
            f"plain score={plain_score:.3f} >= 0.75，与生产短路一致）"
        )
        all_results["docling"] = {
            "skipped": True,
            "reason": "plain_score>=0.75 (P3 switch on)",
        }
    else:
        print("\n[3/3] Docling _parse_with_text_layer ... (可能 60s+)")
        try:
            from rag.pdf_router import has_text_layer
            if not has_text_layer(pdf_path):
                print("   ⚠️ PDF 判定为扫描件，跳过")
                all_results["docling"] = {"skipped": True}
            else:
                t0 = time.perf_counter()
                parser = DoclingParser()
                docling_chunks = parser._parse_with_text_layer(pdf_path)
                t_cost = time.perf_counter() - t0

                report_d = analyze_chunks(docling_chunks, pdf_path)
                report_d["time"] = t_cost

                export_jsonl(docling_chunks, out_dir / f"{pdf_stem}.docling.chunks.jsonl")
                all_results["docling"] = {"chunks": docling_chunks, "report": report_d}

                print(f"   score={report_d['score']:.3f}, 耗时={t_cost:.1f}s, "
                      f"chunks={report_d['n_chunks']}, "
                      f"短chunk={report_d['short_stats']['short_ratio(<=2)']:.1%}, "
                      f"覆盖率={report_d['coverage']:.1%}")
        except Exception as e:
            logger.exception("Docling 失败")
            all_results["docling"] = {"error": str(e)}

    # ── 汇总报告 ──
    print(f"\n{'='*70}")
    print("全量对比")
    print(f"{'='*70}\n")
    print(f"{'候选':<20} {'score':>7} {'耗时':>8} {'chunks':>8} "
          f"{'短chunk%':>10} {'覆盖率':>8} {'缺失页':>8}")
    print(f"{'-'*70}")

    for name, data in all_results.items():
        if "report" not in data:
            print(f"{name:<20} {'N/A':>7} {'N/A':>8} {'N/A':>8} "
                  f"{'N/A':>10} {'N/A':>8} {'N/A':>8}")
            continue
        r = data["report"]
        missing_pages = sum(
            1 for s in r["page_coverage_detail"].values()
            if isinstance(s, str) and s.startswith("missing")
        )
        print(f"{name:<20} {r['score']:>7.3f} {r['time']:>7.1f}s "
              f"{r['n_chunks']:>8} "
              f"{r['short_stats']['short_ratio(<=2)']*100:>9.1f}% "
              f"{r['coverage']*100:>7.1f}% {missing_pages:>8}")

    # ── 导出最优候选的分布报告 ──
    valid = {
        k: v for k, v in all_results.items()
        if "report" in v
    }
    if not valid:
        print("\n[✗] 没有成功的候选")
        sys.exit(1)

    best_name = max(valid, key=lambda k: valid[k]["report"]["score"])
    best = valid[best_name]
    print(f"\n最优候选: {best_name} (score={best['report']['score']:.3f})")

    export_distribution(
        best["chunks"],
        best["report"],
        out_dir / f"{pdf_stem}.chunk_distribution.txt",
    )

    # 保存完整 JSON 报告
    report_path = out_dir / f"{pdf_stem}.quality_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v.get("report", {"error": v.get("error", "skipped")})
             for k, v in all_results.items()},
            f, ensure_ascii=False, indent=2,
        )
    print(f"质量报告 → {report_path}")

    # ── 最终判定 ──
    print(f"\n{'='*70}")
    print("方案可行性判定")
    print(f"{'='*70}\n")

    checks = []
    r = best["report"]

    if r["score"] >= 0.75:
        checks.append(("✓", f"最优候选 score={r['score']:.3f} >= 0.75"))
    else:
        checks.append(("✗", f"最优候选 score={r['score']:.3f} < 0.75"))

    if r["short_stats"]["short_ratio(<=2)"] < 0.10:
        checks.append(("✓", f"短 chunk 占比 {r['short_stats']['short_ratio(<=2)']:.1%} < 10%"))
    else:
        checks.append(("✗", f"短 chunk 占比 {r['short_stats']['short_ratio(<=2)']:.1%} >= 10%"))

    if r["coverage"] >= 0.85:
        checks.append(("✓", f"覆盖率 {r['coverage']:.1%} >= 85%"))
    else:
        checks.append(("✗", f"覆盖率 {r['coverage']:.1%} < 85%"))

    missing_pages = [
        p for p, s in r["page_coverage_detail"].items()
        if isinstance(s, str) and s.startswith("missing")
    ]
    if not missing_pages:
        checks.append(("✓", "无完全缺失的页"))
    else:
        checks.append(("✗", f"{len(missing_pages)} 页完全缺失: {missing_pages[:10]}"))

    for mark, msg in checks:
        print(f"  {mark} {msg}")

    all_pass = all(m != "✗" for m, _ in checks)
    print(f"\n{'='*70}")
    if all_pass:
        print("✅ 方案可行")
    else:
        print("⚠️ 方案需调整")
    print(f"{'='*70}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()