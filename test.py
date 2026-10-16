"""
测试 Docling 层级解析 + RAG 知识库能力。

前置条件：
    1. 后端服务已启动（默认 http://localhost:8000）
    2. IBM.pdf 和天梯 PDF 已经通过 /api/rag/add_file 上传入库

运行：
    python test.py
"""
import sys
import json
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000"


# ═══════════════════════════════════════════════════════════
# 1. 健康检查
# ═══════════════════════════════════════════════════════════
def check_health():
    try:
        r = requests.get(f"{BASE_URL}/api/rag/stats", timeout=10)
        r.raise_for_status()
        stats = r.json()
        print(f"[✓] RAG 服务正常，统计信息: {stats}")
        return True
    except Exception as e:
        print(f"[✗] RAG 服务不可用: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# 2. 上传 PDF（如果还没入库）
# ═══════════════════════════════════════════════════════════
def upload_pdf(filepath: str):
    import os
    if not os.path.exists(filepath):
        print(f"[!] 文件不存在: {filepath}")
        return False
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/rag/add_file",
            files={"file": (filename, f, "application/pdf")},
            timeout=600,
        )
    result = r.json()
    print(f"[{result.get('status', '?')}] 上传 {filename}: {result.get('message', '')}")
    return result.get("status") == "success"


# ═══════════════════════════════════════════════════════════
# 3. 纯检索测试（检查 headings 字段）
# ═══════════════════════════════════════════════════════════
def test_search(query: str, top_k: int = 3):
    print(f"  请求中...", flush=True)
    try:
        r = requests.post(
            f"{BASE_URL}/api/rag/search",
            data={
                "query": query,
                "top_k": top_k,
                "rewrite": "false",   # ← 跳过 LLM 改写
            },
            timeout=180,
        )
        print(f"  响应状态: {r.status_code}", flush=True)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"  ✗ 请求超时（180s）")
        return
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
        return

    results = r.json().get("results", [])
    print(f"\n{'─' * 60}")
    print(f"Query: {query}")
    print(f"{'─' * 60}")

    if not results:
        print("  ⚠️ 无检索结果")
        return

    for i, doc in enumerate(results, 1):
        headings = doc.get("headings", [])
        pages = doc.get("page_numbers", [])
        text = doc.get("text", "")

        print(f"\n  [{i}] headings={headings}")
        print(f"      pages={pages}")
        print(f"      text={text[:120]!r}")

        if len(headings) >= 2:
            print(f"      ✓ 层级路径保留（{len(headings)} 层）")
        elif len(headings) == 1:
            print(f"      ⚠️ 只有 1 层 heading")
        else:
            print(f"      ✗ 无 heading")

# ═══════════════════════════════════════════════════════════
# 4. 完整问答测试（SSE 流式）
# ═══════════════════════════════════════════════════════════
def test_rag_query(query: str, top_k: int = 3):
    print(f"\n{'═' * 60}")
    print(f"RAG 问答: {query}")
    print(f"{'═' * 60}")

    with requests.post(
        f"{BASE_URL}/api/rag/query/stream",
        data={"query": query, "top_k": top_k},
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        answer_parts = []
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "resources":
                print(f"  [资源] {len(data.get('resources', []))} 个")
            elif "text" in data:
                answer_parts.append(data["text"])
                print(data["text"], end="", flush=True)

    print()
    return "".join(answer_parts)


# ═══════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not check_health():
        sys.exit(1)

    # ── 上传测试 PDF（如果还没入库）──
    import os
    if os.path.exists("IBM.pdf"):
        upload_pdf("IBM.pdf")
    if os.path.exists("第十届“中国高校计算机大赛-团体程序设计天梯赛”全国总决赛获奖名单.pdf"):
        upload_pdf("第十届“中国高校计算机大赛-团体程序设计天梯赛”全国总决赛获奖名单.pdf")

    # ── 纯检索测试（重点看 headings 字段）──
    print("\n\n" + "=" * 60)
    print("纯检索测试（检查 headings 是否保留）")
    print("=" * 60)

    # IBM 测试问题
    test_search("IBM 在二战期间做了什么")
    test_search("SABRE 系统属于哪个时期")

    # 天梯测试问题
    test_search("杨嘉福拿了多少分")
    test_search("全国团队二等奖有哪些学校")

    # ── 完整问答测试 ──
    print("\n\n" + "=" * 60)
    print("完整问答测试（SSE 流式）")
    print("=" * 60)

    test_rag_query("IBM 在二战期间做了什么")
    test_rag_query("杨嘉福拿了多少分")