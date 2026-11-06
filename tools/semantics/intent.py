# tools/semantics/intent.py
"""意图分类的唯一真源。

设计原则：
    1. 所有意图相关的词表集中在这里，其他文件不再各自维护
    2. 分类输出结构化的 IntentSet（标签集合），不是单个字符串
    3. 词表按"互斥性"设计：一个 goal 可以同时命中多个标签，
       但优先级由 IntentSet 的 partial order 决定，不由代码顺序决定
"""

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


# ── 词表（集中维护） ──
_GENERATE_VERBS = frozenset({
    "生成", "做一份", "做一个", "写一份", "写一个", "制作", "设计",
    "起草", "拟一份", "导出", "保存为", "转换", "合成",
    "generate", "create", "make", "build", "design", "draft",
    "export", "convert", "save as",
})

_GENERATE_FORMATS = frozenset({
    "ppt", "pptx", "docx", "word", "excel", "xlsx", "pdf", "csv",
    "文档", "报告", "演示", "幻灯片", "表格",
    "presentation", "slide", "document", "report", "spreadsheet",
})

_MODIFY_VERBS = frozenset({
    "修改", "编辑", "修正", "改正", "更正", "补充", "补齐",
    "增加", "添加", "删除", "移除", "重命名", "移动",
    "修复", "repair", "复制", "copy",
    "fix", "edit", "modify", "correct", "update", "patch",
    "delete", "remove", "rename", "move",
})

_READ_VERBS = frozenset({
    "读取", "查看", "打开", "看看", "显示", "列出", "列一下",
    "read", "view", "open", "show", "list",
})

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
    "早上好", "晚上好", "嗨", "hey",
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


def classify(goal: str) -> IntentSet:
    """把 goal 分类成 IntentSet。单一真源，所有调用方都用这个。"""
    if not goal:
        return IntentSet(primary=INTENT_CHITCHAT)

    g = goal.lower()
    tags = set()
    matched = []

    def _has(words):
        hit = [w for w in words if w in g]
        if hit:
            matched.append(tuple(hit))
        return bool(hit)

    has_generate = _has(_GENERATE_VERBS)
    has_generate_fmt = _has(_GENERATE_FORMATS)
    has_modify = _has(_MODIFY_VERBS)
    has_read = _has(_READ_VERBS)
    has_query = _has(_QUERY_HINTS)
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
    # 1. 有生成动词 + 生成格式 → GENERATE（最高优先级）
    if has_generate and has_generate_fmt:
        tags.add(INTENT_GENERATE)
    # 2. 有修改动词 → MODIFY
    if has_modify:
        tags.add(INTENT_MODIFY)
    # 3. 有读动词 → READ
    if has_read:
        tags.add(INTENT_READ)
    # 4. 有查询词 → QUERY
    if has_query:
        tags.add(INTENT_QUERY)
    # 5. web / survey → QUERY（这两类都是信息查询）
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
    # 8. 寒暄：只在没有任何其他标签时才判 CHITCHAT
    if has_chitchat and not (tags - {INTENT_CHITCHAT}):
        tags.add(INTENT_CHITCHAT)

    # ── tutorial：显式命中优先，"如何 + 动词"无条件优先 ──
    if has_tutorial_explicit:
        tags.add(INTENT_TUTORIAL)
    elif has_tutorial_ambiguous:
        # "如何 + 动词"命中时，tutorial 无条件优先。
        # 原因："如何用 pandas 读取 Excel" 有 READ（"读取"），
        # 但"如何用"明确是问方法，不该被 READ 盖过。
        # "帮我做一个 PPT，主题《如何优雅地拒绝周一早会》"里
        # "如何"后面跟"优雅"（不是动词），不命中正则，不会误判。
        tags.add(INTENT_TUTORIAL)

    # ── 决定 primary ──
    # 顺序即优先级：
    # GENERATE > MODIFY > CODE > DB > READ > QUERY > TUTORIAL > CHITCHAT
    # TUTORIAL 放在 READ/QUERY 之前，因为"如何用 X 读取"应该判 tutorial，
    # 不是 read/query。
    priority = [
        INTENT_GENERATE, INTENT_MODIFY, INTENT_CODE, INTENT_DB,
        INTENT_TUTORIAL, INTENT_READ, INTENT_QUERY,
    ]
    # 特例：代码文件 + 修复动词 → CODE 优先于 MODIFY
    # 原因："修复 calculator/core.py 的 divide 函数" 同时命中 MODIFY 和 CODE，
    # 但语义是"修代码"，不是"文件操作"。直接改 priority 顺序会让
    # "修改 config.py" 也变成 CODE，所以只对 "CODE + FIX" 特判。
    if INTENT_CODE in tags and has_fix:
        primary = INTENT_CODE
    else:
        primary = next((t for t in priority if t in tags), INTENT_CHITCHAT)

    # ── 生成任务的格式副标签：供 _select_shortlist 保底 ──
    secondary = set()
    if INTENT_GENERATE in tags:
        if any(k in g for k in ("ppt", "pptx", "演示", "幻灯片", "presentation", "slide")):
            secondary.add("presentation")
        if any(k in g for k in ("docx", "word", "文档", "报告", "report", "document")):
            secondary.add("document")
        if any(k in g for k in ("excel", "xlsx", "表格", "spreadsheet")):
            secondary.add("spreadsheet")
        if any(k in g for k in ("pdf",)):
            secondary.add("pdf")

    return IntentSet(
        tags=frozenset(tags),
        primary=primary,
        secondary=frozenset(secondary),
        matched=tuple(matched),
    )