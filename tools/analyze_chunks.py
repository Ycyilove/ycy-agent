"""分析 chunks.jsonl，输出统计。

用法：
    python -m tools.analyze_chunks out/tianjie_chunks.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="chunks.jsonl 路径")
    parser.add_argument("--top", type=int, default=10, help="展示前 N 条最长的 chunk")
    args = parser.parse_args()

    path = Path(args.jsonl)
    records = [json.loads(line) for line in path.open(encoding="utf-8")]

    n = len(records)
    print(f"总 chunk 数: {n}")

    # ── chunk 长度分布 ──
    lens = [r["chunk_chars"] for r in records]
    lens_sorted = sorted(lens)
    print(f"\nchunk 长度:")
    print(f"  min={min(lens)}  median={lens_sorted[n//2]}  max={max(lens)}  mean={sum(lens)/n:.1f}")

    # ── 摘要覆盖率 ──
    with_summary = sum(1 for r in records if r["summary"])
    print(f"\n摘要覆盖: {with_summary}/{n} ({with_summary/n*100:.1f}%)")
    print(f"  短于阈值(跳过): {sum(1 for r in records if not r['summary'] and r['chunk_chars'] < 80)}")
    print(f"  调用失败/空返回: {sum(1 for r in records if not r['summary'] and r['chunk_chars'] >= 80)}")

    # ── heading 层级分布 ──
    heading_depths = Counter(len(r["headings"]) for r in records)
    print(f"\nheading 深度分布:")
    for depth in sorted(heading_depths):
        label = f"{depth} 层"
        if depth == 0:
            label += "  ← 无 heading，可能丢失祖先"
        print(f"  {label}: {heading_depths[depth]}")

    # ── heading 内容统计（看是否出现错误的父子关系）──
    heading_pairs = Counter()
    for r in records:
        hs = r["headings"]
        for i in range(len(hs) - 1):
            heading_pairs[(hs[i], hs[i+1])] += 1

    print(f"\nTop {args.top} heading 相邻对:")
    for (h1, h2), cnt in heading_pairs.most_common(args.top):
        print(f"  {cnt:4d}  {h1[:30]!r} > {h2[:30]!r}")

    # ── 最长的 chunk ──
    print(f"\nTop {args.top} 最长 chunk:")
    for r in sorted(records, key=lambda x: -x["chunk_chars"])[:args.top]:
        print(f"  [{r['idx']}] {r['chunk_chars']} 字  摘要={r['summary_chars']} 字")
        print(f"      headings={r['headings']}")
        print(f"      text={r['original_text'][:80]!r}")

    # ── 无摘要的最长 chunk（可能是调用失败）──
    failed = [r for r in records if not r["summary"] and r["chunk_chars"] >= 80]
    if failed:
        print(f"\n无摘要但长度足够（可能调用失败）: {len(failed)} 条")
        for r in failed[:5]:
            print(f"  [{r['idx']}] {r['chunk_chars']} 字  headings={r['headings']}")


if __name__ == "__main__":
    main()