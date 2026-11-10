# tools/semantics/intent.py
"""意图分类的唯一真源。

设计原则：
    1. 词表按"原子"组织：动作原子 + 格式分组，组合出意图，
       而不是穷举整句短语（"做一份"/"做一个"命中率低）。
    2. 礼貌前缀先剥离（"帮我"/"给我"/"能不能"），再匹配动作原子。
    3. 寒暄收窄：去掉寒暄词后剩余 < 5 字符才算 chitchat，
       避免"闹钟设置在明天9:00谢谢"被"谢谢"误判。
    4. primary 决定优先级：GENERATE > MODIFY > CODE > DB >
       TUTORIAL > READ > QUERY > CHITCHAT；都没命中则 unknown。
    5. 所有意图相关的词表集中在这里，其他文件不再各自维护。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet


# ── 意图标签 ──
INTENT_GENERATE = "generate"        # 生成文件（docx/xlsx/pptx/pdf）
INTENT_MODIFY = "modify"            # 修改已有文件
INTENT_READ = "read"                # 读取文件内容
INTENT_QUERY = "query"              # 信息查询（知识库/网页）
INTENT_DB = "db"                    # 数据库操作
INTENT_CODE = "code"                # 代码修复/执行
INTENT_TUTORIAL = "tutorial"        # 问方法（不要求执行）
INTENT_CHITCHAT = "chitchat"        # 闲聊
INTENT_UNKNOWN = "unknown"          # 词表未命中（不是闲聊）
INTENT_ANALYZE = "analyze"          # 分析/总结/解读类请求

# ── 词表版本（改词表时 bump，用于缓存 key 失效） ──
WORDLIST_VERSION = "2026-11-07-v1"


# ── 动作原子（单字/词根，组合格式词使用） ──
# ── 词表从 vocab 唯一真源取（B5-b） ──
from .vocab import (
    GEN_ACTION_ATOMS as _GEN_ACTION_ATOMS,
    EN_GEN_ACTION_ROOTS as _EN_GEN_ACTION_ROOTS,
    MODIFY_ACTION_ATOMS as _MODIFY_ACTION_ATOMS,
    READ_ACTION_ATOMS as _READ_ACTION_ATOMS,
    QUERY_ACTION_ATOMS as _QUERY_ACTION_ATOMS,
    ANALYZE_ACTION_ATOMS as _ANALYZE_ACTION_ATOMS,
    FORMAT_GROUPS as _FORMAT_GROUPS,
    GENERATE_FORMATS as _GENERATE_FORMATS,
    POLITE_PREFIXES as _POLITE_PREFIXES,
    QUERY_HINTS as _QUERY_HINTS,
    TUTORIAL_HINTS as _TUTORIAL_HINTS,
    TUTORIAL_AMBIGUOUS_PATTERNS as _TUTORIAL_AMBIGUOUS_PATTERNS,
    DB_HINTS as _DB_HINTS,
    CODE_HINTS as _CODE_HINTS,
    WEB_HINTS as _WEB_HINTS,
    CHITCHAT_HINTS as _CHITCHAT_HINTS,
    SURVEY_HINTS as _SURVEY_HINTS,
    FIX_VERBS as _FIX_VERBS,
)

# 兼容旧名（下游可能 from tools.semantics.intent import _GENERATE_VERBS 等）
_GENERATE_VERBS = _GEN_ACTION_ATOMS
_MODIFY_VERBS = _MODIFY_ACTION_ATOMS
_READ_VERBS = _READ_ACTION_ATOMS


@dataclass(frozen=True)
class IntentSet:
    """一次分类的输出。标签之间是"或"关系，但优先级由 primary 决定。"""
    tags: FrozenSet[str] = field(default_factory=frozenset)
    # 主标签：最高优先级的那个，供 requires_tool_call 等使用
    primary: str = ""
    # 副标签：辅助信息，供 _select_shortlist 的声明式保底用
    secondary: FrozenSet[str] = field(default_factory=frozenset)
    # 副标签按"格式词首次出现位置"排序：((fmt, pos), ...)，第 1 个是主格式
    # 老代码构造 IntentSet(tags=..., primary=..., secondary=...) 时本字段为 ()
    secondary_ordered: tuple = ()
    # 调试用
    matched: tuple = ()


# ─────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────

def _strip_polite_prefix(goal: str) -> str:
    """剥离礼貌前缀（"帮我"/"给我"/"能不能"）。

    返回剥离后的字符串；剥离时如果前缀里含动作原子，调用方可以通过
    对原始 goal 再做一次动作匹配来补偿。
    """
    g = (goal or "").strip()
    for p in sorted(_POLITE_PREFIXES, key=len, reverse=True):
        if g.startswith(p):
            return g[len(p):].strip()
    return g


def _match_action(
    atoms: frozenset, text: str, en_roots: tuple = (),
) -> list:
    """匹配动作原子，返回命中的原子列表。

    en_roots：英文词根表，只对 generate 传 _EN_GEN_ACTION_ROOTS。
    modify/read/query 的英文原子是整词（edit/update/read/...），
    不需要词根匹配；如果对所有词表都用 _EN_GEN_ACTION_ROOTS，
    "create" 会被误匹配成 modify/read/query。
    """
    hit = [a for a in atoms if a in text]
    if not hit and en_roots and any(
        ch.isascii() and ch.isalpha() for ch in text
    ):
        words = re.findall(r"[a-z]+", text.lower())
        for w in words:
            if any(w.startswith(root) for root in en_roots):
                hit.append(w)
                break
    return hit


def _match_formats(text: str) -> list:
    """匹配格式分组，返回命中的分组名列表（无序）。"""
    hit = []
    for fmt, kws in _FORMAT_GROUPS.items():
        if any(k in text for k in kws):
            hit.append(fmt)
    return hit


def _match_formats_with_position(text: str) -> list:
    """匹配格式分组，返回 [(fmt, first_pos), ...]，按 first_pos 升序。

    第 1 个是"主格式"（用户最先提到的），供 _fallback_decide / _select_shortlist
    区分主次（"生成 PPT，另外提 Excel" 主是 presentation，次是 spreadsheet）。

    注意：text 先 lower()，因为 _FORMAT_GROUPS 里的关键词都是小写。
    """
    if not text:
        return []
    text_lower = text.lower()
    hits = []
    for fmt, kws in _FORMAT_GROUPS.items():
        positions = [text_lower.find(k) for k in kws if k in text_lower]
        if positions:
            hits.append((fmt, min(positions)))
    hits.sort(key=lambda x: (x[1], x[0]))
    return hits


def _is_pure_chitchat(text: str) -> bool:
    """去掉寒暄词后剩余 < 5 字符，才算真闲聊。"""
    stripped = text
    for w in _CHITCHAT_HINTS:
        stripped = stripped.replace(w, "")
    stripped = stripped.strip(" ，。,.\t\n").strip()
    return len(stripped) < 5


# ─────────────────────────────────────────────────────────────
# 主分类函数
# ─────────────────────────────────────────────────────────────

def _strip_quoted_content(goal: str) -> str:
    """剥离 goal 里的引用内容，只保留指令部分。

    规则：
        1. 《...》 里长度 > 50 的内容剥离（短书名号保留，
           如《测试报告》是格式词）
        2. "..." / “...” / '...' 里长度 > 50 的内容剥离
        3. 冒号后长度 > 200 的连续片段剥离
    返回：指令部分的文本；如果剥离后为空，返回原 goal。
    """
    if not goal:
        return goal

    g = goal

    # 1. 书名号：长度 > 50 的剥离，短的保留
    def _strip_book(m):
        inner = m.group(1)
        return "" if len(inner) > 50 else m.group(0)
    g = re.sub(r"《([^》]*)》", _strip_book, g)

    # 2. 引号：长度 > 50 的剥离
    def _strip_quote(m):
        inner = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
        return "" if len(inner) > 50 else m.group(0)
    g = re.sub(r"“([^”]*)”", _strip_quote, g)
    g = re.sub(r'"([^"]*)"', _strip_quote, g)
    g = re.sub(r"'([^']*)'", _strip_quote, g)

    # 3. 冒号后超长片段：`XXX：{5000字}` 里冒号后的内容
    #    如果冒号后长度 > 200，剥离冒号后到最后一个标点之前的内容，
    #    保留最后一个标点之后的短尾巴（指令部分）。
    #    例："《文章名》：{5000字}，请你分析这篇文章"
    #    → "《文章名》 请你分析这篇文章"（保留"请你分析这篇文章"）
    m = re.match(r"^(.{1,50})[：:](.{200,})$", g, re.S)
    if m:
        prefix = m.group(1)
        long_content = m.group(2)
        last_punct = -1
        for i in range(len(long_content) - 1, -1, -1):
            if long_content[i] in "。！？.!?，,":
                last_punct = i
                break
        if last_punct > 0:
            tail = long_content[last_punct + 1:].strip()
            g = prefix + " " + tail
        else:
            g = prefix

    g = g.strip()
    # 剥离后为空 → 返回原 goal（避免全剥离导致 classify 失效）
    return g or goal

def _split_goal_segments(goal: str, max_seg_len: int = 200) -> list:
    """把 goal 切成 <= max_seg_len 字符的小段。

    切分点优先级：强标点 > 逗号（段 >= 100 字符才切）> 换行 > 硬切。
    返回 [(seg_text, start, end)]。
    """
    if not goal:
        return []
    if len(goal) <= max_seg_len:
        return [(goal, 0, len(goal))]

    _STRONG_PUNCT = "。！？；.!?;"
    _WEAK_PUNCT = "，,"
    segments = []
    seg_start = 0
    i = 0
    while i < len(goal):
        ch = goal[i]
        if ch in _STRONG_PUNCT:
            seg = goal[seg_start:i + 1].strip()
            if seg:
                segments.append((seg, seg_start, i + 1))
            seg_start = i + 1
        elif ch in _WEAK_PUNCT:
            if i - seg_start >= 100:
                seg = goal[seg_start:i + 1].strip()
                if seg:
                    segments.append((seg, seg_start, i + 1))
                seg_start = i + 1
        elif ch == "\n":
            seg = goal[seg_start:i + 1].strip()
            if seg:
                segments.append((seg, seg_start, i + 1))
            seg_start = i + 1
        i += 1
    if seg_start < len(goal):
        seg = goal[seg_start:].strip()
        if seg:
            segments.append((seg, seg_start, len(goal)))

    final = []
    for seg, s, e in segments:
        if len(seg) <= max_seg_len:
            final.append((seg, s, e))
        else:
            for k in range(0, len(seg), max_seg_len):
                sub = seg[k:k + max_seg_len].strip()
                if sub:
                    final.append((sub, s + k, s + k + len(sub)))
    return final


def _classify_segments(goal: str) -> dict:
    """分段分类，聚合出"指令段"的意图。

    规则：
        - 每段 classify（短段走 _classify_single，不会无限递归）
        - 段里 tags 非空 → 指令段
        - 段里 tags 空且长度 > 100 → 内容段（引用）
        - 聚合：所有指令段的 tags 取并集；primary 取最高优先级；
          secondary 取并集。
        - 如果没有指令段（全是内容段），primary=unknown。
    """
    segments = _split_goal_segments(goal)
    instruction_segments = []
    content_segments = []

    for seg, s, e in segments:
        if not seg:
            continue
        seg_intent = _classify_single(seg)
        if seg_intent.tags:
            instruction_segments.append((seg, seg_intent))
        elif len(seg) > 100:
            content_segments.append((seg, s, e))

    if not instruction_segments:
        return {
            "instruction_segments": [],
            "content_segments": content_segments,
            "aggregated_tags": set(),
            "aggregated_primary": INTENT_UNKNOWN,
            "aggregated_secondary": set(),
            "aggregated_secondary_ordered": (),
        }

    all_tags = set()
    all_secondary = set()
    all_secondary_ordered = []
    for seg, it in instruction_segments:
        all_tags.update(it.tags)
        all_secondary.update(it.secondary)
        # 位置排序取第一个指令段的 ordered（主段决定主次）
        if not all_secondary_ordered and getattr(it, "secondary_ordered", ()):
            all_secondary_ordered = list(it.secondary_ordered)

    priority = [
        INTENT_GENERATE, INTENT_MODIFY, INTENT_CODE, INTENT_DB,
        INTENT_TUTORIAL, INTENT_ANALYZE, INTENT_READ, INTENT_QUERY,
        INTENT_CHITCHAT,
    ]
    primary = next((t for t in priority if t in all_tags), INTENT_UNKNOWN)

    return {
        "instruction_segments": instruction_segments,
        "content_segments": content_segments,
        "aggregated_tags": all_tags,
        "aggregated_primary": primary,
        "aggregated_secondary": all_secondary,
        "aggregated_secondary_ordered": tuple(all_secondary_ordered),
    }


def classify(goal: str) -> IntentSet:
    """意图分类入口：短 goal 单段分类，长 goal 分段聚合。"""
    if not goal:
        return IntentSet(primary=INTENT_CHITCHAT)
    if len(goal) <= 200:
        return _classify_single(goal)
    seg = _classify_segments(goal)
    return IntentSet(
        tags=frozenset(seg["aggregated_tags"]),
        primary=seg["aggregated_primary"],
        secondary=frozenset(seg["aggregated_secondary"]),
        secondary_ordered=tuple(seg.get("aggregated_secondary_ordered", ())),
        matched=(),
    )

def _classify_single(goal: str) -> IntentSet:
    """把 goal 分类成 IntentSet。单一真源，所有调用方都用这个。"""
    if not goal:
        return IntentSet(primary=INTENT_CHITCHAT)

    # 剥离引用内容，只对指令部分做词表匹配
    goal_for_match = _strip_quoted_content(goal)
    g = goal_for_match.lower()
    g_stripped = _strip_polite_prefix(g)

    tags = set()
    matched = []

    def _has(words, text=None):
        src = text if text is not None else g
        hit = [w for w in words if w in src]
        if hit:
            matched.append(tuple(hit))
        return bool(hit)

    # ── 动作匹配 ──
    # 对原始 goal 和剥离前缀后的 goal 都匹配，避免"帮我弄个"剥离"帮我弄"
    # 后丢了"弄"这个动作
    _gen_hits = (
        _match_action(_GEN_ACTION_ATOMS, g_stripped, _EN_GEN_ACTION_ROOTS)
        or _match_action(_GEN_ACTION_ATOMS, g, _EN_GEN_ACTION_ROOTS)
    )
    _mod_hits = (
        _match_action(_MODIFY_ACTION_ATOMS, g_stripped)
        or _match_action(_MODIFY_ACTION_ATOMS, g)
    )
    _read_hits = _match_action(_READ_ACTION_ATOMS, g_stripped)
    _query_hits = _match_action(_QUERY_ACTION_ATOMS, g_stripped)
    _analyze_hits = _match_action(_ANALYZE_ACTION_ATOMS, g_stripped)
    
    has_generate = bool(_gen_hits)
    has_modify = bool(_mod_hits)
    has_read = bool(_read_hits)
    has_query = bool(_query_hits)
    has_analyze = bool(_analyze_hits)

    if _gen_hits:
        matched.append(("generate",) + tuple(_gen_hits))
    if _mod_hits:
        matched.append(("modify",) + tuple(_mod_hits))
    if _read_hits:
        matched.append(("read",) + tuple(_read_hits))
    if _query_hits:
        matched.append(("query",) + tuple(_query_hits))
    # 分析类：有 analyze 动作 → ANALYZE
    if has_analyze:
        tags.add(INTENT_ANALYZE)

    # ── 格式匹配 ──
    _fmt_hits = _match_formats(g_stripped) or _match_formats(g)
    has_formats = bool(_fmt_hits)
    if _fmt_hits:
        matched.append(("formats",) + tuple(_fmt_hits))

    # ── 其他信号 ──
    has_query_hints = _has(_QUERY_HINTS)
    has_tutorial_explicit = _has(_TUTORIAL_HINTS)
    has_tutorial_ambiguous = any(
        p.search(g) for p in _TUTORIAL_AMBIGUOUS_PATTERNS
    )
    has_db = _has(_DB_HINTS)
    has_code = _has(_CODE_HINTS)
    has_web = _has(_WEB_HINTS)
    has_chitchat = _has(_CHITCHAT_HINTS)
    has_survey = _has(_SURVEY_HINTS)
    has_fix = _has(_FIX_VERBS)

    # ── 优先级规则（互斥性靠这里保证） ──
    # 1. 有生成动作 + 格式 → GENERATE
    if has_generate and has_formats:
        tags.add(INTENT_GENERATE)
    # 2. 有修改动作 → MODIFY
    if has_modify:
        tags.add(INTENT_MODIFY)
    # 3. 有读动作 → READ
    if has_read:
        tags.add(INTENT_READ)
    # 4. 有查询词 → QUERY
    if has_query_hints or has_query:
        tags.add(INTENT_QUERY)
    # 5. web / survey → QUERY
    if has_web:
        tags.add(INTENT_QUERY)
    if has_survey:
        tags.add(INTENT_QUERY)
    # 6. 数据库 / 代码
    if has_db:
        tags.add(INTENT_DB)
    if has_code:
        tags.add(INTENT_CODE)
    # 7. 修复动词 → MODIFY
    if has_fix:
        tags.add(INTENT_MODIFY)
    # 8. 寒暄：去掉寒暄词后剩余 < 5 字符，才算真闲聊。
    #    避免"闹钟设置在明天9:00谢谢"被"谢谢"误判成 chitchat。
    if has_chitchat and not (tags - {INTENT_CHITCHAT}):
        if _is_pure_chitchat(g_stripped):
            tags.add(INTENT_CHITCHAT)

    # ── tutorial：显式命中优先，"如何 + 动词"无条件优先 ──
    if has_tutorial_explicit:
        tags.add(INTENT_TUTORIAL)
    elif has_tutorial_ambiguous:
        tags.add(INTENT_TUTORIAL)

    # ── 决定 primary ──
    # 顺序即优先级：
    # GENERATE > MODIFY > CODE > DB > READ > QUERY > TUTORIAL > CHITCHAT
    priority = [
        INTENT_GENERATE, INTENT_MODIFY, INTENT_CODE, INTENT_DB,
        INTENT_TUTORIAL, INTENT_ANALYZE, INTENT_READ, INTENT_QUERY,
    ]
    # 特例：代码文件 + 修复动词 → CODE 优先于 MODIFY
    if INTENT_CODE in tags and has_fix:
        primary = INTENT_CODE
    else:
        primary = next((t for t in priority if t in tags), None)

    # tags 为空时：区分"真闲聊"和"词表未命中"
    if primary is None:
        if INTENT_CHITCHAT in tags:
            primary = INTENT_CHITCHAT
        else:
            primary = INTENT_UNKNOWN

    # ── 格式副标签：所有意图都给，供 _select_shortlist 排序用。
    # 之前只在 generate 时给，导致"读 PDF"的 read_pdf_text
    # 在排序时拿不到 pdf 加分，排 read_csv 后面。 ──
    fmt_ordered = _match_formats_with_position(g_stripped) or _match_formats_with_position(g)
    secondary = {fmt for fmt, _pos in fmt_ordered} or set(_fmt_hits)

    return IntentSet(
        tags=frozenset(tags),
        primary=primary,
        secondary=frozenset(secondary),
        secondary_ordered=tuple(fmt_ordered),
        matched=tuple(matched),
    )


__all__ = [
    "IntentSet",
    "classify",
    "WORDLIST_VERSION",
    "INTENT_GENERATE",
    "INTENT_MODIFY",
    "INTENT_READ",
    "INTENT_QUERY",
    "INTENT_DB",
    "INTENT_CODE",
    "INTENT_TUTORIAL",
    "INTENT_CHITCHAT",
    "INTENT_UNKNOWN",
    # 兼容旧代码：外部可能 import 这些扁平词表
    "_GENERATE_VERBS",
    "_GENERATE_FORMATS",
    "_MODIFY_VERBS",
    "_READ_VERBS",
    "_QUERY_HINTS",
    "_split_goal_segments",
    "_classify_segments",
    "_classify_single",
    "_match_formats_with_position",
]
