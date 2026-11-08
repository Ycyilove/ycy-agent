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
_GEN_ACTION_ATOMS = frozenset({
    # 中文单字
    "生", "做", "写", "弄", "整", "搞", "出", "来", "制", "建",
    "创", "拟", "撰", "编", "起", "草", "导", "存", "转", "合",
    # 英文单词/短语
    "generate", "create", "make", "build", "draft", "write",
    "compose", "assemble", "prepare", "produce", "render",
    "export", "convert", "save",
    "put together", "put-together",
})

_EN_GEN_ACTION_ROOTS = (
    "creat", "gener", "mak", "build", "draft", "writ", "compos",
    "assembl", "prepar", "produc", "render", "export", "convert",
)

_MODIFY_ACTION_ATOMS = frozenset({
    "改", "修", "编", "调", "优", "美", "润", "补", "删", "移",
    "增", "加", "换", "替", "重命名", "重排", "改改",
    # 批处理/转换类（"批量处理"、"转换"是 modify 语义）
    # 注意：不加裸"处理"/"process"，太宽（"处理这个PDF"可能想读/分析）
    "批量", "批处理", "转换", "拆分", "合并", "压缩",
    "edit", "update", "modify", "patch", "fix", "repair",
    "correct", "replace", "insert", "append", "delete", "remove",
    "move", "rename", "optimize", "polish", "refine",
    "batch", "convert", "split", "merge", "compress",
})

_READ_ACTION_ATOMS = frozenset({
    # "查" 留给 _QUERY_ACTION_ATOMS（"查一下"是检索，不是读文件）
    "读", "看", "列", "显", "打", "统", "计", "算", "览",
    "read", "view", "open", "show", "list", "display",
    "count", "calculate", "compute", "preview",
})

_QUERY_ACTION_ATOMS = frozenset({
    "搜", "找", "查", "问", "询", "检",
    "search", "find", "query", "lookup", "retrieve", "ask",
})

_ANALYZE_ACTION_ATOMS = frozenset({
    "分析", "总结", "概括", "解读", "解析", "提炼", "归纳",
    "评价", "评估", "审查", "审阅", "点评", "赏析",
    "analyze", "analyse", "summarize", "summarise",
    "interpret", "review", "evaluate", "assess", "critique",
})

# ── 格式分组（按领域分层，组合动作原子使用） ──
_FORMAT_GROUPS = {
    "presentation": frozenset({
        "ppt", "pptx", "presentation", "slide", "slides", "deck",
        "幻灯片", "演示", "演示文稿", "汇报材料", "宣讲",
    }),
    "document": frozenset({
        "word", "docx", "document", "文档", "报告", "论文", "文章",
        "essay", "paper", "thesis", "简历", "resume", "cv",
        "合同", "contract", "通知", "notice", "发票", "invoice",
        "证书", "certificate", "周报", "简报", "纪要", "minutes",
        "信", "letter", "邮件", "email",
    }),
    "spreadsheet": frozenset({
        "excel", "xlsx", "spreadsheet", "sheet", "表格", "报表",
        "账单", "明细", "清单", "统计表", "看板", "dashboard",
        "csv", "流水",
    }),
    "pdf": frozenset({"pdf"}),
    "table": frozenset({"table", "tabular"}),
    "markdown": frozenset({"markdown", "md"}),
}

# 兼容旧代码：扁平格式词表
_GENERATE_FORMATS = frozenset().union(*_FORMAT_GROUPS.values())


# ── 礼貌前缀（剥离后匹配动作原子） ──
_POLITE_PREFIXES = (
    "帮我", "帮我做", "帮我弄", "帮我整", "帮我搞", "帮我出",
    "给我", "给我做", "给我来", "给我整",
    "能不能", "可不可以", "麻烦", "请", "劳驾",
    "please", "could you", "can you", "would you",
)

# 兼容旧代码：扁平生成动词表（保留 export 给外部 import）
_GENERATE_VERBS = _GEN_ACTION_ATOMS
_MODIFY_VERBS = _MODIFY_ACTION_ATOMS
_READ_VERBS = _READ_ACTION_ATOMS


_QUERY_HINTS = frozenset({
    "是什么", "有哪些", "多少", "什么", "介绍", "讲了什么",
    "统计", "计算", "搜索", "查找",
    "what", "how many", "count", "calculate", "search", "find",
})

_TUTORIAL_HINTS = frozenset({
    "教程", "教我", "怎么用", "命令是",
    "tutorial", "how to", "command is",
})

# "如何/怎么做" 是模糊词：单独出现不足以判 tutorial。
# 只有"如何 + 动词"（如何用/如何读/如何写/如何实现...）才是真的问方法。
# "如何"后面跟非动词（如"如何优雅地拒绝"）可能是内容，不是教程。
_TUTORIAL_AMBIGUOUS_PATTERNS = (
    re.compile(
        r"如何\s*(用|读|写|实现|做|弄|搞|配置|安装|使用|调用|运行|执行|设置|连接|导入|导出|解析|转换|生成)"
    ),
    re.compile(r"怎么做\s*(一|个|出|到|成|好)"),
)

_DB_HINTS = frozenset({"数据库", "sql", "mysql", "查询表", "database"})

_CODE_HINTS = frozenset({
    "代码", "bug", "test", "tests", "重构", "函数",
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".sql", ".sh",
    "code", "refactor", "function",
})

_WEB_HINTS = frozenset({
    "网页", "页面", "网站", "网址", "链接", "榜单", "排行榜",
    "新闻", "最新", "实时", "动态", "热搜", "维基", "官网",
    "web", "page", "website", "url", "link", "news", "latest",
    "wiki",
})

_CHITCHAT_HINTS = frozenset({
    "你好", "hi", "hello", "谢谢", "thanks", "在吗", "测试",
    "早上好", "晚上好", "嗨", "hey", "拜拜", "再见",
})

_SURVEY_HINTS = frozenset({
    "这些文档", "所有文档", "各个文档", "每个文档", "几个文档",
    "多少文档", "分别", "盘点", "速览", "概览", "知识库",
})

_FIX_VERBS = frozenset({"修复", "fix", "repair", "patch", "correct"})


@dataclass(frozen=True)
class IntentSet:
    """一次分类的输出。标签之间是"或"关系，但优先级由 primary 决定。"""
    tags: FrozenSet[str] = field(default_factory=frozenset)
    # 主标签：最高优先级的那个，供 requires_tool_call 等使用
    primary: str = ""
    # 副标签：辅助信息，供 _select_shortlist 的声明式保底用
    secondary: FrozenSet[str] = field(default_factory=frozenset)
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
    """匹配格式分组，返回命中的分组名列表。"""
    hit = []
    for fmt, kws in _FORMAT_GROUPS.items():
        if any(k in text for k in kws):
            hit.append(fmt)
    return hit


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
        }

    all_tags = set()
    all_secondary = set()
    for seg, it in instruction_segments:
        all_tags.update(it.tags)
        all_secondary.update(it.secondary)

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
    secondary = set(_fmt_hits)

    return IntentSet(
        tags=frozenset(tags),
        primary=primary,
        secondary=frozenset(secondary),
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
]