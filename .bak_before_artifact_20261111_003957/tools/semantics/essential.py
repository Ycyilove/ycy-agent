"""essential_when 自动推导。

设计：essential_when 不再手动维护，从两个来源派生：
    1. capability_tags 里的"意图词"（generate/read/modify/...）
    2. name + description 的词根推断

词表约束：essential_when 的取值空间必须和
tools.semantics.intent 的 tags/secondary 对齐，否则
_select_shortlist 的交集判断永远为空。
"""

from __future__ import annotations

import re
from typing import Tuple


# ── essential_when 的合法词表（必须和 intent.py 对齐） ──
# 动作词：对应 intent.tags
ESSENTIAL_ACTIONS = frozenset({
    "generate", "read", "modify", "query", "delete", "db", "convert",
})

# 格式词：对应 intent.secondary
ESSENTIAL_FORMATS = frozenset({
    "presentation", "document", "spreadsheet", "pdf", "table", "office"
})

_ESSENTIAL_VOCAB = ESSENTIAL_ACTIONS | ESSENTIAL_FORMATS


# ── 动作词根 ──
_ACTION_PATTERNS = (
    ("generate", re.compile(
        r"(?:^|[^a-z0-9])("
        r"create|generate|write|export|render|save|build|make|produce|"
        r"merge|compose|assemble|draft|prepare"
        r")(?:[^a-z0-9]|$)",
        re.I,
    )),
    ("read", re.compile(
        r"(?:^|[^a-z0-9])("
        r"read|get|list|show|open|load|fetch|extract|parse|"
        r"ocr|preview|diff|verify|check|inspect|stat|info|status"
        r")(?:[^a-z0-9]|$)",
        re.I,
    )),
    ("modify", re.compile(
        r"(?:^|[^a-z0-9])("
        r"edit|update|modify|patch|replace|append|insert|"
        r"compress|redact|sign|scrub|protect|fill|manipulate|"
        r"batch|process"
        r")(?:[^a-z0-9]|$)",
        re.I,
    )),
    ("query", re.compile(
        r"(?:^|[^a-z0-9])(search|query|find|lookup|retrieve|research|discover)(?:[^a-z0-9]|$)",
        re.I,
    )),
    ("delete", re.compile(
        r"(?:^|[^a-z0-9])(delete|remove|drop|purge|truncate|erase|wipe|clean|scrub)(?:[^a-z0-9]|$)",
        re.I,
    )),
    ("convert", re.compile(
        r"(?:^|[^a-z0-9])(convert|transform)(?:[^a-z0-9]|$)|-to-|_to_",
        re.I,
    )),
)

# ── 格式词根 ──
_FORMAT_PATTERNS = (
    ("presentation", re.compile(
        r"\b(pptx?|powerpoint|presentation|slide|slides|deck)\b", re.I,
    )),
    ("document", re.compile(
        r"\b(docx?|word|document|report|book|epub)\b", re.I,
    )),
    ("spreadsheet", re.compile(
        r"\b(xlsx?|excel|spreadsheet|sheet|csv)\b", re.I,
    )),
    ("pdf", re.compile(r"\bpdf\b", re.I)),
    ("table", re.compile(r"\b(table|tabular)\b", re.I)),
    ("office", re.compile(
        r"(?:^|[^a-z0-9])(office|docx?|xlsx?|pptx?)(?:[^a-z0-9]|$)", re.I,
    )),
)

# ── 排除词：这些工具名不参与"生成"推断 ──
# 比如 pdf_to_excel 含 pdf+excel，但它是转换工具，不是生成工具
_CONVERT_PATTERN = re.compile(
    r"(?:^|[^a-z0-9])(convert|transform)(?:[^a-z0-9]|$)|-to-|_to_",
    re.I,
)


def essential_from_capability_tags(capability_tags):
    if not capability_tags:
        return ()
    tags = {str(t).strip().lower() for t in capability_tags if t}
    actions = tuple(sorted(tags & ESSENTIAL_ACTIONS))
    formats = tuple(sorted(tags & ESSENTIAL_FORMATS))
    # 没有动作词时，格式词单独出现没有意义（如 convert_to_pdf 只有 pdf）
    if not actions:
        return ()
    return actions + formats


def infer_essential_when(
    tool_name: str,
    description: str = "",
    capability_tags: Tuple[str, ...] = (),
) -> Tuple[str, ...]:
    """从工具名 + description + capability_tags 推断 essential_when。

    优先级：
        1. capability_tags 里有合法词 → 直接用（作者显式声明最可靠）
        2. name + description 正则推断
        3. 都没命中 → 返回 ()（不强制保留）
    """
    # 1. 优先 capability_tags
    from_caps = essential_from_capability_tags(capability_tags)
    if from_caps:
        return from_caps

    # 2. 从 name + description 推断
    # 把 _ 替换成 -，让 \bsearch\b 能匹配 search_content（_ 是单词字符，
    # 原样时 search 和 _ 之间没有词边界）
    name = (tool_name or "").lower().replace("_", "-")
    desc = (description or "").lower()
    haystack = f"{name} {desc}"

    # 转换类工具不参与"生成"推断
    is_convert = bool(_CONVERT_PATTERN.search(name))

    actions = []
    for action, pat in _ACTION_PATTERNS:
        if action == "generate" and is_convert:
            continue
        if pat.search(haystack):
            actions.append(action)

    if not actions and any(
        k in name for k in ("read", "get", "list", "show", "preview")
    ):
        actions.append("read")

    # 转换类工具：格式词只从 name 取，不从 description 取
    # （convert_to_pdf 的 description "Convert a document to PDF" 里的
    #   document 是输入格式，不是输出格式，不应进 essential_when）
    formats = []
    if is_convert:
        for fmt, pat in _FORMAT_PATTERNS:
            if pat.search(name):
                formats.append(fmt)
    else:
        for fmt, pat in _FORMAT_PATTERNS:
            if pat.search(haystack):
                formats.append(fmt)

    # 没有动作词时，格式词单独出现没有意义（如 office_batch 只推 office）
    if not actions:
        return ()

    seen = set()
    result = []
    for t in actions + formats:
        if t in _ESSENTIAL_VOCAB and t not in seen:
            seen.add(t)
            result.append(t)

    return tuple(result)


__all__ = [
    "ESSENTIAL_ACTIONS",
    "ESSENTIAL_FORMATS",
    "essential_from_capability_tags",
    "infer_essential_when",
]