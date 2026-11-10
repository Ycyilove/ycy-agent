"""通用完成审计：从 goal 抽取 requirements，对照 state 证据判定。

设计原则（对齐 Codex 的 completion audit）：
    - 不区分 task_type。任何任务都走同一套"要求 → 证据"映射。
    - requirements 抽取对任何任务一视同仁。
    - 证据来自 state 的通用字段（read_paths / written_paths /
      verification_evidence / succeeded_tool_names），不按任务类型分支。
    - 规则优先，规则判不了再交给 LLM（当前版本保守：未解决视为未满足）。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .task_types import GapKind


# ─────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────

@dataclass
class Requirement:
    id: str
    text: str
    kind: str                       # read | modify | verify | fetch | answer
    evidence_hint: str = ""
    satisfied: bool = False
    evidence: str = ""


@dataclass
class AuditResult:
    satisfied: bool
    requirements: List[Requirement] = field(default_factory=list)
    gap: str = ""
    gap_kind: GapKind = GapKind.NONE
    next_hint: str = ""


# ─────────────────────────────────────────────────────────────
# 要求抽取
# ─────────────────────────────────────────────────────────────

# kind 判定：按优先级匹配，命中即定
_KIND_RULES = (
    ("verify", re.compile(
        r"(测试|tests?\b|验证|确保.{0,6}通过|make.{0,6}pass|run.{0,6}test|检查.{0,6}通过)",
        re.I,
    )),
    ("modify", re.compile(
        r"(修复|修改|改正|更正|补充|补齐|增加|添加|编辑|写入|写一|重写|重构|"
        r"移动|重命名|删除|移除|复制|拷贝|整理|合并|拆分|转换|生成|导出|保存|"
        r"fix|repair|patch|correct|edit|modify|write|rewrite|refactor|"
        r"move|rename|delete|remove|copy|merge|split|convert|generate|export|save)",
        re.I,
    )),
    ("fetch", re.compile(
        r"(搜索|检索|查找|查一下|查询|查下|搜一下|抓取|下载|"
        r"网页|网站|网址|链接|新闻|最新|实时|"
        r"search|fetch|crawl|scrape|download|web|url|link|news|latest)",
        re.I,
    )),
    ("read", re.compile(
        r"(读取|查看|打开|显示|列出|列出内容|看看|统计|计算|介绍|讲了什么|"
        r"是什么|有哪些|多少|"
        r"read|view|open|show|list|count|calculate|what)",
        re.I,
    )),
)

# 子目标切分：按换行、分号、编号列表、并列/转折连接词拆。
# 连接词前后常是不同动作（"修复…并确保 tests 通过" → modify + verify），
# 不拆会导致整句被单一 kind 规则吃掉。
_SUBGOAL_SPLIT = re.compile(
    r"[\n;；]"
    r"|(?:^|\s)(?:\d+[.、)）]|[-*•])"
    # 并列/转折连接词：避免误拆"合并/并列/并发/并行/并非"
    r"|，\s*(?:并且|并确保|并要|并需|同时|然后|之后|随后|再|以及)"
    r"|。\s*(?:并且|并确保|同时|然后|之后)"
    r"|(?:,|\s)+(?:and|then|after that)\s+",
    re.I,
)
# 过短的片段不当子目标（避免"请" "帮忙" 被拆出来）
_MIN_SUBGOAL_LEN = 4


def _split_subgoals(goal: str) -> List[str]:
    if not goal:
        return []
    parts = _SUBGOAL_SPLIT.split(goal)
    out = []
    for p in parts:
        p = (p or "").strip()
        if len(p) >= _MIN_SUBGOAL_LEN:
            out.append(p)
    return out or [goal.strip()]


def _classify_kind(text: str) -> str:
    for kind, pat in _KIND_RULES:
        if pat.search(text):
            return kind
    return "answer"


def _classify_kinds_multi(text: str) -> List[str]:
    """一个子目标可能同时命中多个 kind（如"修复…并确保 tests 通过"）。

    返回命中的 kind 列表（按 _KIND_RULES 顺序）；无命中返回 ["answer"]。
    """
    hits = [kind for kind, pat in _KIND_RULES if pat.search(text)]
    return hits or ["answer"]


def extract_requirements(goal: str) -> List[Requirement]:
    """从 goal 抽取要求。通用：任何任务都走这里。

    每个子目标可产出多个 Requirement（多 kind 命中），去重后返回。
    """
    reqs: List[Requirement] = []
    seen: set = set()
    for sg in _split_subgoals(goal):
        sg = sg.strip()
        if not sg:
            continue
        for kind in _classify_kinds_multi(sg):
            key = (kind, sg)
            if key in seen:
                continue
            seen.add(key)
            hint = {
                "read": "read 类工具成功 + 目标文件在 read_paths 里",
                "fetch": "search/fetch 类工具成功",
                "modify": "写类工具成功 + 目标文件在 written_paths 里",
                "verify": "验证类工具成功且输出为通过",
                "answer": "已有足够信息可直接回答",
            }.get(kind, "")
            reqs.append(Requirement(
                id=f"r{len(reqs)}", text=sg, kind=kind, evidence_hint=hint,
            ))
    return reqs


# ─────────────────────────────────────────────────────────────
# 多文档覆盖（从 core.py 迁入，供 read 类要求使用）
# ─────────────────────────────────────────────────────────────

_MULTI_HINTS = (
    "这些文档", "所有文档", "各个文档", "每个文档",
    "几个文档", "多少文档", "分别", "盘点", "速览", "概览",
)


def _norm_filename(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return os.path.basename(name.replace("\\", "/").strip())


def extract_target_files(goal: str, state: Any) -> set:
    if not any(h in (goal or "") for h in _MULTI_HINTS):
        return set()
    target = getattr(state, "_target_files_cache", None) if state else None
    if target:
        return {_norm_filename(f) for f in target if f}
    return set()


# ─────────────────────────────────────────────────────────────
# 证据采集
# ─────────────────────────────────────────────────────────────

_FETCH_TOOL_HINTS = (
    "fetch", "http_get", "fetch_json", "search", "crawl", "tavily",
)


def _is_fetch_tool(name: str) -> bool:
    low = (name or "").lower()
    return any(h in low for h in _FETCH_TOOL_HINTS)


def collect_evidence(state: Any) -> Dict[str, Any]:
    succeeded = dict(getattr(state, "succeeded_tool_names", {}) or {})
    fetch_tools = {n for n in succeeded if _is_fetch_tool(n)}
    return {
        "history": list(getattr(state, "history", []) or []),
        "succeeded_tools": succeeded,
        "failed_calls": dict(getattr(state, "failed_calls", {}) or {}),
        "read_paths": set(getattr(state, "read_paths", set()) or set()),
        "written_paths": set(getattr(state, "written_paths", set()) or set()),
        "verification": getattr(state, "verification_evidence", None),
        "fetch_tools_succeeded": fetch_tools,
    }


# ─────────────────────────────────────────────────────────────
# 审计主逻辑
# ─────────────────────────────────────────────────────────────

_KIND_TO_GAP = {
    "read": GapKind.MISSING_CONTENT,
    "fetch": GapKind.MISSING_CONTENT,
    "modify": GapKind.MISSING_ACTION,
    "verify": GapKind.MISSING_VERIFICATION,
    "answer": GapKind.MISSING_CONTENT,
}

_KIND_TO_HINT = {
    "read": "**下一步：读取/检索缺失内容**（read_text_file / search_content / search_knowledge 等）",
    "fetch": "**下一步：调用搜索/抓取工具获取外部信息**",
    "modify": "**下一步：执行写/改操作**（write_file / edit_file 等）",
    "verify": "**下一步：运行验证**（run_tests / 检查类工具）",
    "answer": "**下一步：补充信息或直接回答**",
}

# ── run_audit 里 missing 排序 ──
_GAP_PRIORITY = {"modify": 0, "read": 1, "fetch": 1, "verify": 2, "answer": 3}

def run_audit(
    goal: str,
    state: Any,
    evidence: Optional[Dict[str, Any]] = None,
) -> AuditResult:
    """逐条要求对照证据。规则优先；未解决的要求视为未满足。"""
    if evidence is None:
        evidence = collect_evidence(state)

    reqs = extract_requirements(goal)
    if not reqs:
        return AuditResult(satisfied=True, requirements=[])

    # 证据类工具成功是否为空：用于 answer 类兜底
    any_success = bool(evidence["succeeded_tools"])

    # ── 规则层 ──
    for r in reqs:
        if r.kind == "modify":
            if evidence["written_paths"]:
                r.satisfied = True
                r.evidence = f"written: {sorted(evidence['written_paths'])}"
            else:
                # 兜底：写类工具成功但没记录到路径（如返回 bytes、
                # 或路径参数名不在 _OUTPUT_PATH_KEYS 里）。
                # 只要写类工具成功执行过，就视为 modify 满足，
                # 避免因路径采集不全导致 done 被反复拒绝。
                write_succeeded = [
                    name for name in evidence["succeeded_tools"]
                    if _looks_like_write_tool(name)
                ]
                if write_succeeded:
                    r.satisfied = True
                    r.evidence = (
                        f"写类工具已成功（未记录路径）: "
                        f"{write_succeeded[:5]}"
                    )
                else:
                    # 写尝试过但失败：不视为满足，但记录
                    write_failed = any(
                        _looks_like_write_tool(name)
                        for name in evidence["failed_calls"]
                    )
                    if write_failed:
                        r.evidence = "写操作尝试过但失败"

        elif r.kind == "read":
            target = extract_target_files(goal, state)
            if target:
                covered = {_norm_filename(p) for p in evidence["read_paths"]}
                missing = target - covered
                r.satisfied = not missing
                r.evidence = (
                    f"read {len(covered)}/{len(target)}; "
                    f"missing={sorted(missing)}"
                )
            elif evidence["read_paths"]:
                r.satisfied = True
                r.evidence = f"read: {sorted(evidence['read_paths'])}"

        elif r.kind == "fetch":
            if evidence["fetch_tools_succeeded"]:
                r.satisfied = True
                r.evidence = (
                    f"fetch tools: {sorted(evidence['fetch_tools_succeeded'])}"
                )

        elif r.kind == "verify":
            v = evidence["verification"]
            if isinstance(v, dict):
                r.satisfied = bool(v.get("success"))
                r.evidence = str(v)[:200]

        elif r.kind == "answer":
            r.satisfied = any_success

    missing = [r for r in reqs if not r.satisfied]
    if not missing:
        return AuditResult(satisfied=True, requirements=reqs)
    missing.sort(key=lambda r: _GAP_PRIORITY.get(r.kind, 9))
    first = missing[0]

    return AuditResult(
        satisfied=False,
        requirements=reqs,
        gap=f"要求「{first.text}」尚无证据（期望：{first.evidence_hint or first.kind}）",
        gap_kind=_KIND_TO_GAP.get(first.kind, GapKind.NONE),
        next_hint=_KIND_TO_HINT.get(first.kind, ""),
    )


_WRITE_TOOL_HINTS = ("write", "edit", "create", "mkdir", "rename", "move",
                     "delete", "remove", "save", "patch", "update")


def _looks_like_write_tool(name: str) -> bool:
    low = (name or "").lower()
    return any(h in low for h in _WRITE_TOOL_HINTS)