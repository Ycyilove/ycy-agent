"""Tool call 相关的数据类。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolCallRequest:
    tool_name: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    user_confirmation_needed: bool


@dataclass
class ToolCallResult:
    tool_name: str
    success: bool
    result: Any = None
    error: str = None
    execution_time: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentAnalysis:
    needs_tool: bool
    intent_type: str
    confidence: float
    suggested_tool: str = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
