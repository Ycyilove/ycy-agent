"""Multi-step ReAct orchestrator (thin wrapper).

实际实现已移至 tools/orchestration/。
本模块保持向后兼容的 import 路径。
"""

from .orchestration import (
    AgentOrchestrator,
    OrchestratorStep,
    ORCHESTRATOR_PROMPT,
    OrchestratorState,
    requires_tool_call,
    classify_request,
    extract_intent,
    route_intent,
    INTENT_TO_TAG,
)
from .orchestration.state import params_key as _params_key


__all__ = [
    "AgentOrchestrator",
    "OrchestratorStep",
    "ORCHESTRATOR_PROMPT",
    "OrchestratorState",
    "requires_tool_call",
    "classify_request",
    "extract_intent",
    "route_intent",
    "INTENT_TO_TAG",
    "_params_key",
]