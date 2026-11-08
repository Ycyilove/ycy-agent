"""意图分类稳定性检查。

用法：
    python devtools/check_intent_stability.py record   # 记录基线
    python devtools/check_intent_stability.py compare  # 比对基线

基线存到 devtools/intent_baseline.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "devtools" / "intent_baseline.json"

# 20 个覆盖各分支的 goal
GOALS = [
    "你好",
    "hi",
    "谢谢",
    "今天天气怎么样",
    "帮我生成一份 PPT，主题是季度总结",
    "做一个 Excel 表格，统计销售额",
    "写一个 Word 文档，介绍 AI",
    "生成一份 PDF 报告",
    "把 report.docx 转成 PDF",
    "修改 config.py 的 DATA_DIR",
    "修复 calculator/core.py 的 divide 函数，并确保 tests 通过",
    "帮我重构这个函数",
    "读取 data.csv 的内容",
    "列出当前目录的文件",
    "查看 report.pdf 讲了什么",
    "搜索最近的 AI 新闻",
    "查一下维基百科上维尼修斯的当前状态",
    "知识库里记录了哪些获得最佳女队称号的学校",
    "这些文档分别讲了什么",
    "怎么用 Python 读取 Excel",
]


def _snapshot() -> list[dict]:
    from tools.semantics import classify_intent
    from tools.intent_frame.normalizer import IntentNormalizer

    normalizer = IntentNormalizer()
    out = []
    for goal in GOALS:
        intent = classify_intent(goal)
        frame = normalizer.normalize(goal)
        out.append({
            "goal": goal,
            "intent": {
                "primary": intent.primary,
                "tags": sorted(intent.tags),
                "secondary": sorted(intent.secondary),
            },
            "frame": {
                "domain": frame.domain,
                "action": frame.action,
                "freshness": frame.freshness,
            },
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
        print(f"    intent.old: {d['old']['intent']}")
        print(f"    intent.new: {d['new']['intent']}")
        print(f"    frame.old:  {d['old']['frame']}")
        print(f"    frame.new:  {d['new']['frame']}")
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