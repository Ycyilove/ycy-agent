"""Multi-step orchestration subpackage.

- core:           AgentOrchestrator 主类
- prompt:         ORCHESTRATOR_PROMPT
- state:          OrchestratorState
- validator:      requires_tool_call（策略 B）
- intent_router:  route_intent（策略 D）
"""

from .core import AgentOrchestrator, OrchestratorStep
from .prompt import ORCHESTRATOR_PROMPT
from .state import OrchestratorState
from .validator import requires_tool_call, classify_request
from .intent_router import extract_intent, route_intent, INTENT_TO_TAG

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
]