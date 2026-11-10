"""召回稳定性检查。

用法：
    python devtools/check_recall_stability.py record
    python devtools/check_recall_stability.py compare

基线存到 devtools/recall_baseline.json。
覆盖：
    - tool_shortlister（goal -> top-k 工具）
    - mcp_intent_index（goal -> top-k server）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "devtools" / "recall_baseline.json"

GOALS = [
    "帮我生成一份 PPT，主题是季度总结",
    "做一个 Excel 表格，统计销售额",
    "写一个 Word 文档，介绍 AI",
    "读取 data.csv 的内容",
    "列出当前目录的文件",
    "搜索最近的 AI 新闻",
    "查一下维基百科上维尼修斯的当前状态",
    "知识库里记录了哪些获得最佳女队称号的学校",
    "怎么用 Python 读取 Excel",
    "修复 calculator/core.py 的 divide 函数",
]


def _snapshot() -> list[dict]:
    """只记录短名单（不依赖 mcp server 在线）。"""
    from tools.semantics.index.strategies.shortlist import shortlist as sl

    class FakeMeta:
        def __init__(self, name, desc, caps, ess):
            self.name = name
            self.description = desc
            self.capability_tags = caps
            self.essential_when = ess

    # 固定一组工具做短名单（不依赖真实 registry）
    tools = {
        "read_pdf_text": FakeMeta("read_pdf_text", "读取 PDF 文本", ("pdf",), ("read", "pdf")),
        "read_docx_text": FakeMeta("read_docx_text", "读取 Word 文本", ("document",), ("read", "document")),
        "read_csv": FakeMeta("read_csv", "读取 CSV", ("csv",), ("read",)),
        "write_docx": FakeMeta("write_docx", "生成 Word 文档", ("document",), ("generate", "document")),
        "write_excel": FakeMeta("write_excel", "生成 Excel", ("spreadsheet",), ("generate", "spreadsheet")),
        "list_files": FakeMeta("list_files", "列出目录文件", ("dir",), ()),
        "search_knowledge": FakeMeta("search_knowledge", "检索知识库", ("knowledge",), ("query",)),
        "http_get": FakeMeta("http_get", "HTTP GET", ("http",), ()),
        "fetch_url": FakeMeta("fetch_url", "下载 URL", ("http", "fetch"), ()),
        "search_and_connect_mcp": FakeMeta("search_and_connect_mcp", "搜索并连接 MCP", ("mcp",), ()),
    }

    out = []
    for goal in GOALS:
        shortlist_result = sl(goal, tools)
        out.append({
            "goal": goal,
            "shortlist": sorted(shortlist_result.keys()) if shortlist_result else [],
        })
    return out


def cmd_record() -> int:
    data = _snapshot()
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"baseline recorded: {BASELINE} ({len(data)} goals)")
    return 0


def cmd_compare() -> int:
    if not BASELINE.exists():
        print(f"baseline not found: {BASELINE}", file=sys.stderr)
        print("run 'record' first", file=sys.stderr)
        return 2
    old = json.loads(BASELINE.read_text(encoding="utf-8"))
    new = _snapshot()
    diffs = []
    for o, n in zip(old, new):
        if o != n:
            diffs.append({"old": o, "new": n})
    if not diffs:
        print(f"OK: {len(new)} goals unchanged")
        return 0
    print(f"DIFF: {len(diffs)}/{len(new)} goals changed", file=sys.stderr)
    for d in diffs:
        print(f"\n  goal: {d['old']['goal']!r}")
        print(f"    old: {d['old']['shortlist']}")
        print(f"    new: {d['new']['shortlist']}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["record", "compare"])
    args = parser.parse_args()
    if args.cmd == "record":
        return cmd_record()
    return cmd_compare()


if __name__ == "__main__":
    sys.exit(main())