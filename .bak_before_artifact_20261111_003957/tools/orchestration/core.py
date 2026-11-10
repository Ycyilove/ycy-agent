"""AgentOrchestrator 主类。

重构后设计：
    - 工具筛选只保留两段：场景过滤 + embedding shortlist
    - 决策权交给 LLM prompt（工具链/前置条件/失败切换都在 prompt 里）
    - 代码只保留安全兜底：环境不可用工具过滤、幂等拦截、完成度校验
"""

from __future__ import annotations

import os
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from infra.debug_log.llm import dump_llm_call

from tools.agent import (
    ToolAgent,
    ToolCallRequest,
    ToolCallResult,
)
from tools.idempotency import classify_tool_idempotency
from tools.semantics import get_intent_tags

from .prompt import ORCHESTRATOR_PROMPT
from .state import OrchestratorState, params_key
from .validator import requires_tool_call
from .task_classifier import classify_task
from .task_types import (
    TaskType, StopReason, AssessResult, GapKind, can_upgrade,
    TERMINAL_REASONS,
)
from .audit import run_audit, collect_evidence
from core.llm.provider import ContextLengthExceeded


logger = logging.getLogger(__name__)

import os as _os

_READ_TOOL_HINTS = (
    "read", "list", "search", "get_file_info", "stat", "fetch", "crawl",
)




# ── 模块级辅助从 helpers.py 取（B8 fix）──
from .helpers import (
    _READ_TOOL_HINTS,
    _is_read_like,
    _norm_filename,
    _format_relative_time,
    _build_structured_history,
    _SucceededSeqView,
)


# ── 4 Mixin 从各自文件 import（B8）──
from .decision import OrchestratorStep, DecisionMixin
from .fill_params import FillParamsMixin
from .assess import AssessMixin
from .run import RunMixin


class AgentOrchestrator(DecisionMixin, FillParamsMixin, AssessMixin, RunMixin):
    _MCP_DISCOVERY_TOOLS = frozenset({"search_and_connect_mcp"})

    # 工具依赖图：key 工具需要的前置工具（B8 重构丢失，补回）
    # 来源：prompt.py::ORCHESTRATOR_PROMPT 的"工具链与前置条件"
    _TOOL_PREREQUISITES = {
        "convert_to_pdf": frozenset({"write_docx", "create_word"}),
        "edit_docx": frozenset({"write_docx", "create_word"}),
        "edit_excel": frozenset({"write_excel", "create_excel"}),
        "edit_pptx": frozenset({"create_pptx"}),
        "merge_pdfs": frozenset({"convert_to_pdf"}),
        "pdf_preview": frozenset({"convert_to_pdf"}),
        "pdf_compress": frozenset({"convert_to_pdf"}),
        "sign_pdf": frozenset({"convert_to_pdf"}),
    }

    # 当前运行环境不可用的工具（Windows 裸机缺 WeasyPrint → render_document 不可用）
    _UNAVAILABLE_TOOLS = frozenset({
        "mcp__office-worker__render_document",
        "mcp__office-worker__pdf_ocr",
        "mcp__office-worker__create_book",
    })

    # 静态配置的 MCP server
    _STATIC_MCP_SERVERS = frozenset({"tavily", "fetch"})

    # 单步工具结果进 state.history 的字符上限
    _HISTORY_LIMITS = {
        "search_knowledge": 3000,
        "knowledge_stats": 1500,
        "list_knowledge_sources": 1500,
        "read_text_file": 1500,
        "read_file_range": 1500,
        "read_pdf_text": 2000,
        "read_docx_text": 2000,
        "read_csv": 1200,
        "search_content": 1200,
        "search_files": 1200,
        "fetch_url": 2000,
        "http_get": 2000,
        "write_file": 300,
        "edit_file": 300,
        "create_directory": 200,
        "move_file": 200,
        "delete_file": 200,
        "run_tests": 2000,
    }
    _DEFAULT_HISTORY_LIMIT = 1000

    def __init__(
        self,
        agent_provider: Callable[[], ToolAgent],
        llm_provider: Callable[[], Any],
        max_steps: int = 10,
        shortlister_provider: Optional[Callable[[], Any]] = None,
    ):
        self._agent_provider = agent_provider
        self._llm_provider = llm_provider
        self._max_steps = max_steps
        self._shortlister_provider = shortlister_provider

        try:
            from backend.config import (
                LLM_PROMPT_BUDGET_CHARS,
                LLM_HISTORY_BUDGET_RATIO,
                LLM_HISTORY_ENTRY_CHARS,
                LLM_COMPACT_TRIGGER_RATIO,
                LLM_COMPACT_KEEP_RECENT,
            )
            self._prompt_budget_chars = LLM_PROMPT_BUDGET_CHARS
            self._history_budget_ratio = LLM_HISTORY_BUDGET_RATIO
            self._history_entry_chars = LLM_HISTORY_ENTRY_CHARS
            self._compact_trigger_ratio = LLM_COMPACT_TRIGGER_RATIO
            self._compact_keep_recent = LLM_COMPACT_KEEP_RECENT
        except Exception:
            self._prompt_budget_chars = 24000
            self._history_budget_ratio = 0.4
            self._history_entry_chars = 1000
            self._compact_trigger_ratio = 0.8
            self._compact_keep_recent = 5


