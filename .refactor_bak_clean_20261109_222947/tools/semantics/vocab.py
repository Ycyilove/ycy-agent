"""意图/领域/动作/格式词表的唯一真源。

所有分类器（intent.py / normalizer.py / tags.py / essential.py）从这里取词，
不再各自维护。改词表只改这一个文件。

组织原则：
    1. 动作原子（单字/词根，组合格式词使用）
    2. 格式分组（按领域分层，组合动作原子使用）
    3. 领域词表（唯一）
    4. 动作词表（唯一）
    5. 信号词（web/db/code/chitchat/survey/fix/query/tutorial）
    6. 礼貌前缀
    7. 领域 → tags.py 用途标签的映射
"""
from __future__ import annotations

import re
from typing import FrozenSet


# ═══════════════════════════════════════════════════════════
# 1. 动作原子
# ═══════════════════════════════════════════════════════════

GEN_ACTION_ATOMS: FrozenSet[str] = frozenset({
    # 中文单字
    "生", "做", "写", "弄", "整", "搞", "出", "来", "制", "建",
    "创", "拟", "撰", "编", "起", "草", "导", "存", "转", "合",
    # 英文单词/短语
    "generate", "create", "make", "build", "draft", "write",
    "compose", "assemble", "prepare", "produce", "render",
    "export", "convert", "save",
    "put together", "put-together",
})

# 兼容旧名
_GENERATE_VERBS = GEN_ACTION_ATOMS

EN_GEN_ACTION_ROOTS: tuple = (
    "creat", "gener", "mak", "build", "draft", "writ", "compos",
    "assembl", "prepar", "produc", "render", "export", "convert",
)

MODIFY_ACTION_ATOMS: FrozenSet[str] = frozenset({
    "改", "修", "编", "调", "优", "美", "润", "补", "删", "移",
    "增", "加", "换", "替", "重命名", "重排", "改改",
    "批量", "批处理", "转换", "拆分", "合并", "压缩",
    "edit", "update", "modify", "patch", "fix", "repair",
    "correct", "replace", "insert", "append", "delete", "remove",
    "move", "rename", "optimize", "polish", "refine",
    "batch", "convert", "split", "merge", "compress",
})

_MODIFY_VERBS = MODIFY_ACTION_ATOMS

READ_ACTION_ATOMS: FrozenSet[str] = frozenset({
    "读", "看", "列", "显", "打", "统", "计", "算", "览",
    "read", "view", "open", "show", "list", "display",
    "count", "calculate", "compute", "preview",
})

_READ_VERBS = READ_ACTION_ATOMS

QUERY_ACTION_ATOMS: FrozenSet[str] = frozenset({
    "搜", "找", "查", "问", "询", "检",
    "search", "find", "query", "lookup", "retrieve", "ask",
})

ANALYZE_ACTION_ATOMS: FrozenSet[str] = frozenset({
    "分析", "总结", "概括", "解读", "解析", "提炼", "归纳",
    "评价", "评估", "审查", "审阅", "点评", "赏析",
    "analyze", "analyse", "summarize", "summarise",
    "interpret", "review", "evaluate", "assess", "critique",
})


# ═══════════════════════════════════════════════════════════
# 2. 格式分组
# ═══════════════════════════════════════════════════════════

FORMAT_GROUPS: dict = {
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

# 兼容旧名：扁平格式词表
GENERATE_FORMATS = frozenset().union(*FORMAT_GROUPS.values())
_GENERATE_FORMATS = GENERATE_FORMATS


# ═══════════════════════════════════════════════════════════
# 3. 领域词表（唯一）
# ═══════════════════════════════════════════════════════════

DOMAIN_KEYWORDS: dict = {
    "news": (
        "新闻", "资讯", "报道", "实时", "最新", "动态", "热搜",
        "news", "latest", "realtime",
    ),
    "sports": (
        "体育", "足球", "篮球", "球员", "比赛", "赛事", "运动员",
        "sports", "football", "basketball", "player", "match",
    ),
    "knowledge": (
        "知识库", "文档", "资料", "概要", "摘要",
        "knowledge", "document", "summary",
    ),
    "file": (
        "文件", "目录", "路径", "csv", "pdf", "docx",
        "file", "directory", "path",
    ),
    "code": (
        "代码", "运行", "执行", "python",
        "code", "run", "execute",
    ),
    "web": (
        "网页", "页面", "网站", "网址", "链接",
        "web", "page", "website", "url",
    ),
}


# ═══════════════════════════════════════════════════════════
# 4. 动作词表（唯一）
# ═══════════════════════════════════════════════════════════

ACTION_KEYWORDS: dict = {
    "search": ("搜索", "查找", "查", "找", "search", "find", "query"),
    "read": ("读取", "查看", "打开", "read", "view", "open"),
    "write": ("创建", "新建", "写入", "保存", "create", "write", "save"),
    "compute": ("计算", "统计", "compute", "calculate", "count"),
    "browse": ("浏览", "抓取", "fetch", "crawl", "extract"),
}

FRESHNESS_KEYWORDS: dict = {
    "realtime": ("实时", "现在", "当前", "此刻", "realtime", "now", "current"),
    "recent": ("最新", "最近", "近期", "这几天", "latest", "recent"),
}


# ═══════════════════════════════════════════════════════════
# 5. 信号词
# ═══════════════════════════════════════════════════════════

QUERY_HINTS: FrozenSet[str] = frozenset({
    "是什么", "有哪些", "多少", "什么", "介绍", "讲了什么",
    "统计", "计算", "搜索", "查找",
    "what", "how many", "count", "calculate", "search", "find",
})

TUTORIAL_HINTS: FrozenSet[str] = frozenset({
    "教程", "教我", "怎么用", "命令是",
    "tutorial", "how to", "command is",
})

TUTORIAL_AMBIGUOUS_PATTERNS: tuple = (
    re.compile(
        r"如何\s*(用|读|写|实现|做|弄|搞|配置|安装|使用|调用|运行|执行|设置|连接|导入|导出|解析|转换|生成)"
    ),
    re.compile(r"怎么做\s*(一|个|出|到|成|好)"),
)

DB_HINTS: FrozenSet[str] = frozenset({"数据库", "sql", "mysql", "查询表", "database"})

CODE_HINTS: FrozenSet[str] = frozenset({
    "代码", "bug", "test", "tests", "重构", "函数",
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".sql", ".sh",
    "code", "refactor", "function",
})

WEB_HINTS: FrozenSet[str] = frozenset({
    "网页", "页面", "网站", "网址", "链接", "榜单", "排行榜",
    "新闻", "最新", "实时", "动态", "热搜", "维基", "官网",
    "web", "page", "website", "url", "link", "news", "latest",
    "wiki",
})

CHITCHAT_HINTS: FrozenSet[str] = frozenset({
    "你好", "hi", "hello", "谢谢", "thanks", "在吗", "测试",
    "早上好", "晚上好", "嗨", "hey", "拜拜", "再见",
})

SURVEY_HINTS: FrozenSet[str] = frozenset({
    "这些文档", "所有文档", "各个文档", "每个文档", "几个文档",
    "多少文档", "分别", "盘点", "速览", "概览", "知识库",
})

FIX_VERBS: FrozenSet[str] = frozenset({"修复", "fix", "repair", "patch", "correct"})


# ═══════════════════════════════════════════════════════════
# 6. 礼貌前缀
# ═══════════════════════════════════════════════════════════

POLITE_PREFIXES: tuple = (
    "帮我", "帮我做", "帮我弄", "帮我整", "帮我搞", "帮我出",
    "给我", "给我做", "给我来", "给我整",
    "能不能", "可不可以", "麻烦", "请", "劳驾",
    "please", "could you", "can you", "would you",
)


# ═══════════════════════════════════════════════════════════
# 7. 领域 → tags.py 用途标签的映射
# ═══════════════════════════════════════════════════════════
# 用于 capability.index::covers() 判断"现有工具是否覆盖某领域"。
# 之前 covers() 里写死 domain 与 tag 的对应，现在集中到这里。

DOMAIN_TO_TAGS: dict = {
    "news": frozenset({"network_read", "search_name"}),
    "sports": frozenset({"network_read", "search_name"}),
    "knowledge": frozenset({"knowledge_read"}),
    "file": frozenset({
        "read_text", "read_binary", "read_dir", "traverse_dir",
        "write_new", "write_modify", "move", "mkdir", "stat",
    }),
    "code": frozenset({"execute_code", "read_text", "write_modify"}),
    "web": frozenset({"network_read", "search_name", "parse"}),
}


__all__ = [
    # 动作原子
    "GEN_ACTION_ATOMS", "EN_GEN_ACTION_ROOTS",
    "MODIFY_ACTION_ATOMS", "READ_ACTION_ATOMS",
    "QUERY_ACTION_ATOMS", "ANALYZE_ACTION_ATOMS",
    # 兼容旧名
    "_GENERATE_VERBS", "_GENERATE_FORMATS",
    "_MODIFY_VERBS", "_READ_VERBS",
    # 格式
    "FORMAT_GROUPS", "GENERATE_FORMATS",
    # 领域 / 动作
    "DOMAIN_KEYWORDS", "ACTION_KEYWORDS", "FRESHNESS_KEYWORDS",
    # 信号词
    "QUERY_HINTS", "TUTORIAL_HINTS", "TUTORIAL_AMBIGUOUS_PATTERNS",
    "DB_HINTS", "CODE_HINTS", "WEB_HINTS",
    "CHITCHAT_HINTS", "SURVEY_HINTS", "FIX_VERBS",
    # 礼貌
    "POLITE_PREFIXES",
    # 领域映射
    "DOMAIN_TO_TAGS",
]