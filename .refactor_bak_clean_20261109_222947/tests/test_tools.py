"""工具链测试：schema 读取 / MCP 归一化 / essential_when 推导。

合并自：
    - test_field_schema.py
    - test_mcp_normalize.py
    - test_essential_infer.py（从脚本改写为 pytest）

不合并：
    - test_scene_leak.py（复刻逻辑，非 import 真实函数，建议删）
"""

from __future__ import annotations

import pytest

from mcp_client.manager import MCPClientManager
from rag.structure.field_classifier import FieldSchema
from tools.semantics import (
    infer_essential_when,
    essential_from_capability_tags,
)


# ═════════════════════════════════════════════════════════════
# 1. read_tool_schema（原 test_field_schema.py）
# ═════════════════════════════════════════════════════════════

class TestFieldSchema:
    def test_empty_schema_returns_syntax_types_only(self):
        s = FieldSchema.empty()
        assert s.resolve("团队", "GZHU-3") == ("code", "value")
        assert s.resolve("姓名", "张三") == ("free_text", "default")
        assert s.resolve("分数", "195") == ("number", "value")

    def test_schema_header_hint_wins(self):
        s = FieldSchema.from_dict({
            "types": {"team": {"header_hints": ["团队"], "is_anchor": True}}
        })
        assert s.resolve("团队", "GZHU-3") == ("team", "header")
        assert s.resolve("队伍", "GZHU-3") == ("code", "value")

    def test_schema_pattern_wins_over_syntax(self):
        s = FieldSchema.from_dict({
            "types": {"team": {"value_patterns": [r"[A-Z]{2,}-\d+"]}}
        })
        assert s.resolve("", "GZHU-3") == ("team", "value")
        assert s.resolve("", "T01") == ("code", "value")


# ═════════════════════════════════════════════════════════════
# 2. _normalize_mcp_result（原 test_mcp_normalize.py）
# ═════════════════════════════════════════════════════════════

class TestMcpNormalize:
    def test_invalid_parameter_not_auth_error(self):
        payload = {
            "isError": False,
            "structuredContent": {"message_detail": "invalid parameter 'x'"},
        }
        out = MCPClientManager._normalize_mcp_result(payload)
        assert out["success"] is True

    def test_missing_api_key_is_auth_error(self):
        payload = {
            "isError": False,
            "structuredContent": {"message_detail": "missing api key"},
        }
        out = MCPClientManager._normalize_mcp_result(payload)
        assert out["success"] is False

    def test_is_error_true(self):
        payload = {"isError": True, "content": [{"type": "text", "text": "boom"}]}
        out = MCPClientManager._normalize_mcp_result(payload)
        assert out["success"] is False
        assert "boom" in out["error"]


# ═════════════════════════════════════════════════════════════
# 3. infer_essential_when（原 test_essential_infer.py，从脚本改写）
# ═════════════════════════════════════════════════════════════

class TestEssentialInfer:
    @pytest.mark.parametrize("name,desc,caps,expected_subset", [
        ("create_pptx", "Create a PowerPoint presentation", (), ["generate", "presentation"]),
        ("create_word", "Create a Word document", (), ["generate", "document"]),
        ("create_excel", "Create an Excel spreadsheet", (), ["generate", "spreadsheet"]),
        ("render_document", "Render a document to PDF", (), ["generate", "pdf"]),
        ("read_pdf_text", "Read text from a PDF file", (), ["read", "pdf"]),
        ("read_docx_text", "Read paragraphs from a Word doc", (), ["read", "document"]),
        ("write_docx", "Write paragraphs to a Word document", (), ["generate", "document"]),
        # 转换工具不应 generate
        ("pdf_to_excel", "Convert PDF tables to Excel", ("convert", "pdf", "excel"), []),
        ("search_web", "Search the web for information", (), ["query"]),
        ("edit_pptx", "Edit an existing PowerPoint", (), ["modify", "presentation"]),
        ("read_office", "Read any office document", ("read", "document"), ["read", "document"]),
        # 空 description（动态发现工具可能没有）
        ("mcp__foo__create_pptx", "", (), ["generate", "presentation"]),
    ])
    def test_infer_essential_when(self, name, desc, caps, expected_subset):
        actual = infer_essential_when(name, desc, caps)
        missing = [e for e in expected_subset if e not in actual]
        assert not missing, (
            f"{name}: expected ⊇ {expected_subset}, got {actual}（缺 {missing}）"
        )

    def test_essential_from_capability_tags(self):
        # 有动作词 + 格式词 → 都保留
        out = essential_from_capability_tags(("generate", "document"))
        assert set(out) >= {"generate", "document"}

        # 只有格式词、没动作词 → 返回空（格式单独出现没意义）
        out = essential_from_capability_tags(("pdf",))
        assert out == ()

        # 空输入
        assert essential_from_capability_tags(()) == ()