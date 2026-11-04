
"""通用审计的快捷验证。不依赖 LLM、不依赖网络。"""
from __future__ import annotations

import pytest

from tools.orchestration.audit import (
    extract_requirements,
    run_audit,
    collect_evidence,
)
from tools.orchestration.task_types import GapKind
from tools.orchestration.state import OrchestratorState


# ── 构造 state 的小工具 ──

def make_state(**kw) -> OrchestratorState:
    st = OrchestratorState(goal=kw.pop("goal", ""))
    for k, v in kw.items():
        setattr(st, k, v)
    return st


# ── 1. 要求抽取 ──

def test_extract_code_fix_requirements():
    goal = (
        "请帮我修复以下问题，并确保 tests/ 下的测试全部通过："
        "1. calculator/core.py 的 divide 函数遇到除零会崩溃 "
        "2. calculator/validators.py 的 validate_positive 逻辑反了 "
        "3. calculator/formatters.py 的 format_result 精度不对 "
        "4. config.py 的 DATA_DIR 硬编码 "
        "5. main.py 调用 divide 时未导入 "
        "修复后请运行 tests/ 验证。"
    )
    reqs = extract_requirements(goal)
    kinds = [r.kind for r in reqs]
    assert "modify" in kinds, kinds
    assert "verify" in kinds, kinds
    # 不应把整段当一个 answer
    assert not all(k == "answer" for k in kinds)


def test_extract_chitchat_requirements():
    reqs = extract_requirements("你好")
    # 无 modify/verify/read/fetch；至少一个 answer
    assert all(r.kind == "answer" for r in reqs)


# ── 2. 读阶段：不满足 modify ──

def test_audit_read_only_blocks_done():
    goal = "修复 calculator/core.py 的 divide 函数，并确保 tests 通过"
    st = make_state(
        goal=goal,
        read_paths={"calculator/core.py"},
        written_paths=set(),
        succeeded_tool_names={"read_text_file": 1},
    )
    res = run_audit(goal, st, evidence=collect_evidence(st))
    assert res.satisfied is False
    assert res.gap_kind == GapKind.MISSING_ACTION, res.gap_kind


# ── 3. 写后无验证：不满足 verify ──

def test_audit_write_without_verify_blocks_done():
    goal = "修复 calculator/core.py 的 divide 函数，并确保 tests 通过"
    st = make_state(
        goal=goal,
        read_paths={"calculator/core.py"},
        written_paths={"calculator/core.py"},
        succeeded_tool_names={"read_text_file": 1, "edit_file": 2},
        verification_evidence=None,
    )
    res = run_audit(goal, st, evidence=collect_evidence(st))
    assert res.satisfied is False
    assert res.gap_kind == GapKind.MISSING_VERIFICATION, res.gap_kind


# ── 4. 写 + 验证通过 → 满足 ──

def test_audit_write_and_verify_pass():
    goal = "修复 calculator/core.py 的 divide 函数，并确保 tests 通过"
    st = make_state(
        goal=goal,
        read_paths={"calculator/core.py"},
        written_paths={"calculator/core.py"},
        succeeded_tool_names={"read_text_file": 1, "edit_file": 2, "run_tests": 3},
        verification_evidence={"tool": "run_tests", "success": True},
    )
    res = run_audit(goal, st, evidence=collect_evidence(st))
    assert res.satisfied is True, res.gap


# ── 5. 网页任务：search 成功但无 extract ──
#    注意：当前 audit 的 fetch 类要求把 search 也算 fetch。
#    若你要"搜索不算完成"，先收紧 audit.py 再启用本测试。

@pytest.mark.xfail(reason="当前 audit 对 search 算 fetch；收紧后应通过")
def test_audit_web_search_only_blocks_done():
    goal = "查一下维基百科上维尼修斯的当前状态"
    st = make_state(
        goal=goal,
        succeeded_tool_names={"mcp__tavily__tavily_search": 1},
    )
    res = run_audit(goal, st, evidence=collect_evidence(st))
    assert res.satisfied is False


# ── 6. 多文档盘点：read_paths 覆盖 target ──

def test_audit_multi_doc_coverage():
    goal = "这些文档分别讲了什么"
    st = make_state(goal=goal)
    object.__setattr__(st, "_target_files_cache", {"a.pdf", "b.pdf"})
    st.read_paths = {"a.pdf"}
    res = run_audit(goal, st, evidence=collect_evidence(st))
    assert res.satisfied is False
    st.read_paths = {"a.pdf", "b.pdf"}
    res = run_audit(goal, st, evidence=collect_evidence(st))
    assert res.satisfied is True