# -*- coding: utf-8 -*-
"""PDF 检索测试：A/B/C/D 四组。"""

import sys
from backend.app import get_local_rag_service

# 测试题：query → 期望关键词（出现在返回 chunk 里）
TEST_CASES = {
    "A 精确人名": [
        ("钱易", ["钱易"]),
        ("汪澄", ["汪澄", "281"]),
        ("朱剑枫", ["朱剑枫", "北京大学"]),
        ("张亲翰", ["张亲翰", "281"]),
        ("管晏如", ["管晏如", "朗润园"]),
    ],
    "B 改述": [
        ("全国高校一等奖有哪些学校", ["北京大学", "一等奖"]),
        ("哪些人拿到了个人冠军奖", ["钱易", "个人冠军"]),
        ("华南师范大学女队有谁", ["华南师范大学", "女队"]),
        ("有哪些学校获得珠峰争鼎", ["珠峰争鼎"]),
    ],
    "C 组合": [
        ("北京大学获奖了吗", ["北京大学"]),
        ("最高分是多少", ["281", "成绩"]),
        ("有哪些女生获奖", ["女"]),
    ],
    "D 跨 chunk": [
        ("钱易和汪澄的成绩对比", ["钱易", "汪澄"]),
        ("北京大学有几支队伍获奖", ["北京大学"]),
    ],
}


def run_group(name, cases, rewrite, use_bm25):
    print(f"\n{'='*60}")
    print(f"## {name} | rewrite={rewrite}, bm25={use_bm25}")
    print(f"{'='*60}")

    rag = get_local_rag_service()
    ok_count = 0

    for query, expected_keywords in cases:
        try:
            results = rag.search(
                query, top_k=10,
                rewrite=rewrite,
                use_bm25=use_bm25,
            )
        except Exception as e:
            print(f"❌ {query!r}: 检索异常 {e}")
            continue

        # 检查返回的 chunk 里有没有期望关键词
        combined_text = " ".join(r.get("text", "") for r in results)
        found = [kw for kw in expected_keywords if kw in combined_text]

        if len(found) == len(expected_keywords):
            mark = "✅"
            ok_count += 1
        elif found:
            mark = "⚠️"
        else:
            mark = "❌"

        # 命中来源
        files = set(r.get("filename", "?") for r in results)
        file_note = f" [来源: {len(files)} 个文件]"

        print(
            f"{mark} {query!r}{file_note}\n"
            f"    期望: {expected_keywords}\n"
            f"    命中: {found}\n"
            f"    返回: {len(results)} 条"
        )

    print(f"\n本组: {ok_count}/{len(cases)} 全命中")


def main():
    print("=" * 60)
    print("RAG 检索测试：A/B/C/D 四组")
    print("=" * 60)

    rag = get_local_rag_service()
    stats = rag.get_stats()
    print(f"\nembedder: {rag.embedding_model}")
    print(f"bm25_store: {rag.bm25_store is not None}")
    print(f"total_files: {stats.get('total_files')}")
    print(f"total_chunks: {stats.get('total_chunks')}")
    for f in stats.get("files", []):
        print(f"  {f['filename']}: {f['chunk_count']} chunks")

    # A 组：不开启任何增强
    run_group("A 精确人名", TEST_CASES["A 精确人名"], rewrite=False, use_bm25=False)

    # B 组：不开启增强
    run_group("B 改述", TEST_CASES["B 改述"], rewrite=False, use_bm25=False)

    # C 组：开启 query 改写
    run_group("C 组合", TEST_CASES["C 组合"], rewrite=True, use_bm25=False)

    # C 组对照：不开启
    run_group("C 组合（无改写）", TEST_CASES["C 组合"], rewrite=False, use_bm25=False)

    # D 组：开启 BM25
    run_group("D 跨 chunk（BM25）", TEST_CASES["D 跨 chunk"], rewrite=False, use_bm25=True)

    # D 组对照：不开启 BM25
    run_group("D 跨 chunk（无 BM25）", TEST_CASES["D 跨 chunk"], rewrite=False, use_bm25=False)


if __name__ == "__main__":
    main()