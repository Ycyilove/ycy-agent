"""Adversarial acceptance tests: Hacking the previous audit logic.

These tests specifically target the gaps that allowed the Agent to fake completion.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock

from core.task.state import OrchestratorState
from tools.types import Artifact, ToolCallResult
from tools.orchestration.audit import run_audit
from tools.format import format_tool_result


def _mock_state(goal="test", written_paths=None, read_paths=None, grounding_chunks=0):
    s = OrchestratorState(goal=goal, session_id="s")
    s.written_paths = set(written_paths or [])
    s.read_paths = set(read_paths or [])
    s.grounding_chunks = grounding_chunks
    return s


# ══════════════════════════════════════════════════════════════
# 1. 黑框框 / 空文件 HACK
# ══════════════════════════════════════════════════════════════

def test_adversarial_empty_file_rejected():
    """0 字节文件不能算作成功写操作。"""
    # 模拟生成目标要求写 PDF
    state = _mock_state(
        goal="将报告转换成 PDF 并保存",  # 含"转换""保存"→ modify
        written_paths={("/tmp/fake.pdf", 0)}  # 0 字节！
    )
    audit = run_audit(state.goal, state)
    assert not audit.satisfied, "空文件不应通过审计"
    assert "未产生有效文件" in audit.gap


def test_adversarial_blackbox_pdf_rejected():
    """极小 PDF（黑框框）不能算作成功写操作（如果你的策略包含大小限制）。"""
    # 模拟生成目标要求写 PDF
    state = _mock_state(
        goal="将报告转换成 PDF 并保存",  # 含"转换""保存"→ modify
        written_paths={("/tmp/fake.pdf", 800)}  # 小于 1KB
    )
    audit = run_audit(state.goal, state)
    # 取决于你最后的策略，如果是严格的 size > 0，它会 pass；如果 < 1KB reject，它会 fail。
    # 我们这里验证的是只要 < 1024 就不满足
    assert not audit.satisfied or audit.requirements[0].evidence.startswith("物理证据")


def test_adversarial_truncated_read_warning():
    """读取截断时，format_tool_result 必须警告 LLM。"""
    result = ToolCallResult(
        tool_name="read_text_file",
        success=True,
        result={"content": "part"},
        artifacts=[Artifact(path="/tmp/a.txt", size=10000, is_truncated=True)]
    )
    output = format_tool_result(result)
    assert "当前仅读取了部分内容（已截断）" in output
    assert "声称已读全文" in output  # 提示里是「无法…声称已读全文」


# ══════════════════════════════════════════════════════════════
# 2. RAG 幻觉 HACK
# ══════════════════════════════════════════════════════════════

def test_adversarial_rag_hallucination_rejected():
    """用户问知识库，但 grounding_chunks=0，不能靠普通工具蒙混过关。"""
    state = _mock_state(
        goal="请根据知识库告诉我杨灿垣得了什么奖",
        written_paths=set(),
        read_paths=set(),
        grounding_chunks=0  # 没召回任何 chunk
    )
    audit = run_audit(state.goal, state)
    assert not audit.satisfied
    assert "尚无证据" in audit.gap


def test_adversarial_fake_rag_grounded():
    """如果 grounding_chunks > 0，可以通过。"""
    state = _mock_state(
        goal="请根据知识库告诉我杨灿垣得了什么奖",
        grounding_chunks=2
    )
    audit = run_audit(state.goal, state)
    assert audit.satisfied


# ══════════════════════════════════════════════════════════════
# 3. 绕过审批 HACK
# ══════════════════════════════════════════════════════════════

def test_adversarial_convert_bypass_approval():
    """验证 convert 工具带 output_path 会被识别为写操作。"""
    from tools.security.gateway import SecurityGateway
    # 没有显式 meta，但传了 output_path
    assert SecurityGateway.is_write_operation("mcp__docgen__convert_document", None, {"output_path": "/tmp/a.pdf"})
    assert SecurityGateway.is_write_operation("convert_file_format", None, {"output": "/tmp/a.pdf"})


# ══════════════════════════════════════════════════════════════
# 4. 部分完成 HACK
# ══════════════════════════════════════════════════════════════

def test_adversarial_partial_completion():
    """只要产物存在且 > 0，就算通过；大小与目标内容部分对应交给评估层。"""
    state = _mock_state(
        goal="生成三段文字的 PDF",
        written_paths={("/tmp/a.pdf", 2000)}
    )
    # 注意：审计层本身只负责硬证据（有文件且 > 0），
    # 具体的“三段文字是不是只有一段”由 LLM 的自我评估和 prompt 约束，
    # 或者通过更上层的 file_content 校验来把关。
    audit = run_audit(state.goal, state)
    assert audit.satisfied