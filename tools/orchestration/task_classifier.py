"""首步任务分类：多维信号打分，不靠单一动词。

与 task_types.TaskType 对应，返回 7 族之一。
分类只看首步；运行时由 core.py 的 _maybe_upgrade_task_type 升级。
"""

from __future__ import annotations

import re
from typing import List, Optional

from .task_types import TaskType


# ── 副作用动词（写/删/移/整理/生成） ──
_SIDE_EFFECT = (
    "创建", "新建", "写入", "写一个", "写个", "编辑", "修改", "改",
    "修复", "修正", "改正", "更正", "补充", "补齐", "增加", "添加",
    "移动", "重命名", "删除", "移除", "复制", "拷贝",
    "整理", "分类", "合并", "拆分", "转换", "生成", "导出", "保存",
    "create", "write", "edit", "modify", "move", "rename", "delete",
    "remove", "copy", "merge", "split", "convert", "generate",
    "export", "save", "fix", "repair", "patch", "correct", "update",
)

# ── 读取动词（查/看/列/统计） ──
_READ = (
    "读取", "查看", "打开", "看", "显示", "列出", "列一下",
    "搜索", "查找", "找一下", "统计", "计算",
    "介绍", "讲了什么", "是什么", "有哪些", "多少", "什么",
    "read", "view", "open", "show", "list", "search", "find",
    "count", "calculate", "what", "how many",
)

# ── 网页信号 ──
_WEB = (
    "网页", "页面", "网站", "网址", "链接", "榜单", "排行榜",
    "新闻", "最新", "实时", "动态", "热搜", "维基", "官网", "wiki",
    "web", "page", "website", "url", "link", "news", "latest",
)

# ── 多文档盘点信号 ──
_SURVEY = (
    "这些文档", "所有文档", "各个文档", "每个文档", "几个文档",
    "多少文档", "分别", "盘点", "速览", "概览", "知识库",
)

# ── 寒暄信号 ──
_CHITCHAT = (
    "你好", "hi", "hello", "谢谢", "thanks", "在吗", "测试",
    "早上好", "晚上好", "嗨", "hey",
)

# ── 生成类动词（比 SIDE_EFFECT 更窄） ──
_GENERATE_VERBS = (
    "生成", "导出", "转换", "保存为", "写一份", "做一份", "合成",
    "generate", "export", "convert", "save as",
)

# ── 代码文件扩展名 ──
_CODE_EXT = re.compile(
    r"\.(py|js|jsx|ts|tsx|java|go|rs|c|cpp|h|hpp|rb|php|sql|sh)\b",
    re.I,
)

# ── 数据文件扩展名 ──
_DATA_EXT = re.compile(
    r"\.(xlsx?|csv|json|docx?|pdf|md|txt)\b",
    re.I,
)

# ── 路径模式（盘符 / 绝对路径 / 带扩展名） ──
_PATH = re.compile(r"[A-Za-z]:[\\/]|/(?:[^/\s]+/)+|\.\w{1,5}\b")

# ── 修复类动词（单独判定 CODE_FIX 用） ──
_FIX_VERBS = ("修复", "fix", "repair", "patch", "correct")


def _any(text: str, words) -> bool:
    return any(w in text for w in words)


def classify_task(goal: str, attachments: Optional[List] = None) -> TaskType:
    """首步任务分类。返回完成判据族。

    判定顺序（命中即返回）：
        chitchat → web_research → doc_survey → code_fix
        → generate → file_op → info_query → 兜底

    Args:
        goal: 用户原始请求
        attachments: 附件列表（用于 generate / file_op 的强信号）
    """
    g = (goal or "").lower().strip()
    n_attach = len(attachments or [])

    if not g and not n_attach:
        return TaskType.CHITCHAT

    has_path = bool(_PATH.search(g))
    has_side = _any(g, _SIDE_EFFECT)
    has_read = _any(g, _READ)
    has_web = _any(g, _WEB)
    has_survey = _any(g, _SURVEY)
    has_data = bool(_DATA_EXT.search(g))
    has_code = bool(_CODE_EXT.search(g))

    # 1. chitchat：短句 + 寒暄词 + 无路径 + 无副作用
    if (
        g
        and len(g) < 20
        and _any(g, _CHITCHAT)
        and not has_path
        and not has_side
    ):
        return TaskType.CHITCHAT

    # 2. 网页检索：有 web 信号优先（后续需要生成由运行时升级）
    if has_web:
        return TaskType.WEB_RESEARCH

    # 3. 多文档盘点
    if has_survey:
        return TaskType.DOC_SURVEY

    # 4. 代码修复：代码文件 + 副作用动词
    if has_code and has_side:
        return TaskType.CODE_FIX
    # 或"修复/跑测试"类明确指令
    if (
        _any(g, _FIX_VERBS)
        and ("test" in g or "tests" in g or has_path)
    ):
        return TaskType.CODE_FIX

    # 5. 内容生成：生成类动词 + 数据文件/附件
    if _any(g, _GENERATE_VERBS) and (has_data or n_attach):
        return TaskType.GENERATE

    # 6. 文件操作：副作用 + 路径/数据文件/附件
    if has_side and (has_path or has_data or n_attach):
        return TaskType.FILE_OP

    # 7. 信息查询
    if has_read:
        return TaskType.INFO_QUERY

    # 8. 兜底
    return TaskType.FILE_OP if has_side else TaskType.INFO_QUERY