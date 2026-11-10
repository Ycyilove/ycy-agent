"""Tool call 相关的数据类。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolCallRequest:
    tool_name: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    user_confirmation_needed: bool


@dataclass
class Artifact:
    """工具产生的物理产物证据（文件、截断信号等）。"""
    path: str = ""
    size: int = 0
    mime: str = ""
    is_truncated: bool = False
    content_preview: str = ""

@dataclass
class ToolCallResult:
    tool_name: str
    success: bool
    result: Any = None
    error: str = None
    execution_time: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Artifact] = field(default_factory=list)


@dataclass
class IntentAnalysis:
    needs_tool: bool
    intent_type: str
    confidence: float
    suggested_tool: str = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
