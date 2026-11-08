"""ToolAgent + IntentRecognizer.

- 引导提取（guidance）已移至 tools/guidance/
- 幂等分类（idempotency）已移至 tools/idempotency/
- 失败分类（failure）已移至 tools/semantics/
- 本模块只保留 ToolAgent 主逻辑 + 规则/LLM 意图识别
- 兼容旧 import：re-export 上述函数
"""

from __future__ import annotations

import logging
import json
from contextvars import ContextVar
from typing import Any, Dict, FrozenSet, List, Optional

from . import registry, get_registry
from .intent import LLMIntentRecognizer, IntentRecognizer
from .types import ToolCallRequest, ToolCallResult, IntentAnalysis

_allowed_tools_var: ContextVar[Optional[FrozenSet[str]]] = ContextVar(
    "tool_agent_allowed_tools", default=None
)


logger = logging.getLogger(__name__)


_MCP_TOOL_PREFIX = "mcp__"


class ToolAgent:
    """工具调用决策和执行。"""

    def __init__(self, registry: registry = None):
        self.registry = registry or get_registry()
        self.rule_intent_recognizer = IntentRecognizer()
        self.llm_intent_recognizer = LLMIntentRecognizer()
        self.tool_descriptions = self.registry.get_tool_descriptions()
        self._mcp_client_manager = None

    def set_mcp_client_manager(self, manager) -> None:
        self._mcp_client_manager = manager
        logger.info(
            "[dsh][mcp] ToolAgent bound to MCPClientManager: %r",
            type(manager).__name__,
        )

    @property
    def _allowed_tools(self) -> Optional[FrozenSet[str]]:
        return _allowed_tools_var.get()

    def set_available_tools(self, tool_names: List[str]) -> None:
        if not tool_names:
            _allowed_tools_var.set(None)
        else:
            _allowed_tools_var.set(frozenset(tool_names))
        logger.info(f"[Skills] Allowed tools set to: {len(tool_names)} tools")

    def clear_available_tools(self) -> None:
        _allowed_tools_var.set(None)

    @staticmethod
    def _is_mcp_tool_name(tool_name: str) -> bool:
        return (
            isinstance(tool_name, str)
            and tool_name.startswith(_MCP_TOOL_PREFIX)
            and "__" in tool_name[len(_MCP_TOOL_PREFIX):]
        )

    def register_dynamic_tool(
        self, name, description, parameters=None, handler=None,
        input_schema=None, origin="builtin", danger_level="safe",
        idempotent=False, essential_when=(),
        capability_tags=(),
    ):
        from tools.security.trust import assign_trust_level
        trust_level = assign_trust_level(
            name, origin, self._mcp_client_manager,
        )
        logger.info(
            "[dsh][mcp] register tool: %s (origin=%s, trust=L%d, "
            "essential_when=%r)",
            name, origin, trust_level, tuple(essential_when or ()),
        )
        self.registry.register_callable(
            handler, name=name, description=description,
            parameters=parameters, input_schema=input_schema,
            origin=origin, danger_level=danger_level,
            idempotent=idempotent, trust_level=trust_level,
            essential_when=tuple(essential_when or ()),
            capability_tags=tuple(capability_tags or ()),
        )
        self.tool_descriptions = self.registry.get_tool_descriptions()
        self.llm_intent_recognizer._tool_descriptions_cache = None
        return handler

    def unregister_dynamic_tool(self, name: str) -> bool:
        metadata = self.registry.get_metadata(name)
        if metadata is None or metadata.origin != "mcp":
            return False
        removed = self.registry.unregister(name)
        self.tool_descriptions = self.registry.get_tool_descriptions()
        self.llm_intent_recognizer._tool_descriptions_cache = None
        return removed

    def list_all_tools(self, allowed_danger_levels: Optional[set] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, meta in self.registry.get_all_metadata().items():
            if self._allowed_tools is not None and name not in self._allowed_tools:
                continue
            if allowed_danger_levels is not None and meta.danger_level not in allowed_danger_levels:
                continue
            result[name] = meta
        return result

    def analyze_intent(self, user_message: str, use_llm: bool = True) -> IntentAnalysis:
        if use_llm:
            llm_result = self.llm_intent_recognizer.analyze(user_message, self.registry)
            if llm_result.suggested_tool:
                if self._allowed_tools is None or llm_result.suggested_tool in self._allowed_tools:
                    return llm_result
        rule_result = self.rule_intent_recognizer.analyze(user_message)
        if rule_result.suggested_tool:
            if self._allowed_tools is None or rule_result.suggested_tool in self._allowed_tools:
                return rule_result
        return IntentAnalysis(
            needs_tool=False, intent_type="general_conversation",
            confidence=0.0, reasoning="未识别到工具调用需求",
        )

    def prepare_tool_call(self, intent: IntentAnalysis) -> Optional[ToolCallRequest]:
        if not intent.needs_tool or not intent.suggested_tool:
            return None
        tool_func = self.registry.get_tool(intent.suggested_tool)
        if not tool_func:
            return None
        metadata = self.registry.get_metadata(intent.suggested_tool)
        danger_level = metadata.danger_level if metadata else "safe"
        return ToolCallRequest(
            tool_name=intent.suggested_tool,
            parameters=intent.parameters,
            confidence=intent.confidence,
            reasoning=intent.reasoning,
            user_confirmation_needed=danger_level in ["medium", "high"],
        )

    async def execute_tool_async(self, tool_call: ToolCallRequest) -> ToolCallResult:
        import inspect
        import time
        start_time = time.time()
        tool_name = tool_call.tool_name
        tool_func = self.registry.get_tool(tool_name)
        params = tool_call.parameters or {}

        from tools.security.gateway import get_gateway, get_security_context
        ctx = get_security_context()
        meta = self.registry.get_metadata(tool_name)
        if ctx is not None:
            decision = get_gateway().check(tool_name, params, meta, ctx)
            if not decision.allowed:
                logger.warning(
                    "[dsh][security] blocked %s: %s", tool_name, decision.reason
                )
                return ToolCallResult(
                    tool_name=tool_name, success=False,
                    error=f"安全网关拒绝: {decision.reason}",
                    execution_time=time.time() - start_time,
                    parameters=params,
                )

        # ── 新增：L2 或写操作必须 verify 审批 ──
        if meta is not None:
            trust = getattr(meta, "trust_level", 0)
            is_write = get_gateway().is_write_operation(tool_name, meta)
            if trust == 2 or is_write:
                approval_id = params.get("__approval_id__")
                if not approval_id:
                    return ToolCallResult(
                        tool_name=tool_name, success=False,
                        error="需要审批但缺少 __approval_id__",
                        execution_time=time.time() - start_time,
                        parameters=params,
                    )
                from tools.security.approval import get_approval_store
                if not get_approval_store().verify(approval_id, tool_name, params):
                    return ToolCallResult(
                        tool_name=tool_name, success=False,
                        error=(
                            f"审批校验失败：approval_id={approval_id} "
                            "已过期/已消费/参数不匹配"
                        ),
                        execution_time=time.time() - start_time,
                        parameters=params,
                    )

        if tool_func is None and self._is_mcp_tool_name(tool_name):
            mcp_client = self._mcp_client_manager
            if mcp_client is None:
                return ToolCallResult(
                    tool_name=tool_name, success=False,
                    error=f"工具 {tool_name} 未注册且 MCPClientManager 未注入。",
                    execution_time=time.time() - start_time,
                    parameters=params,
                )
            logger.info("[dsh][mcp] dynamic route: %s params=%r", tool_name, params)
            try:
                result = await mcp_client.call_tool(tool_name, params)
                if isinstance(result, dict):
                    sc = result.get("structuredContent")
                    if isinstance(sc, dict) and sc.get("error"):
                        success = False
                        error_message = str(sc.get("error"))
                    elif "success" in result:
                        success = bool(result["success"])
                        error_message = None if success else result.get("error")
                    elif result.get("error"):
                        success = False
                        error_message = str(result["error"])
                    else:
                        success = True
                        error_message = None
                else:
                    success = True
                    error_message = None

                if isinstance(result, dict):
                    blocks = result.get("content")
                    if isinstance(blocks, list) and blocks:
                        first = blocks[0]
                        if isinstance(first, dict) and first.get("type") == "text":
                            try:
                                inner = json.loads(first.get("text", "{}"))
                                if (
                                    isinstance(inner, dict)
                                    and inner.get("status") == "error"
                                ):
                                    success = False
                                    error_message = (
                                        inner.get("error")
                                        or "tool returned error status"
                                    )
                                    logger.info(
                                        "[dsh][mcp] tool %s returned status=error: %s",
                                        tool_name, error_message[:200],
                                    )
                            except (json.JSONDecodeError, TypeError):
                                pass

                return ToolCallResult(
                    tool_name=tool_name, success=success, result=result,
                    error=error_message, execution_time=time.time() - start_time,
                    parameters=params,
                )
            except Exception as error:
                logger.exception("[dsh][mcp] dynamic route failed: %s", tool_name)
                hint = ""
                try:
                    if mcp_client is not None:
                        available = [spec.public_name for spec in mcp_client.tools.values()]
                        if available:
                            hint = (
                                "\n\n⚠️ 工具名可能拼错了。当前可用的 MCP 工具列表"
                                "（**请用以下确切名称重试**）：\n"
                                + "\n".join(f"  - {t}" for t in available[:30])
                            )
                except Exception:
                    logger.exception("failed to collect hint")
                return ToolCallResult(
                    tool_name=tool_name, success=False,
                    error=str(error) + hint,
                    execution_time=time.time() - start_time,
                    parameters=params,
                )

        if tool_func is None:
            return ToolCallResult(
                tool_name=tool_name, success=False,
                error=f"工具不存在: {tool_name}",
                execution_time=time.time() - start_time,
                parameters=params,
            )

        try:
            result = tool_func(**params)
            if inspect.isawaitable(result):
                result = await result
            execution_time = time.time() - start_time
            if isinstance(result, dict):
                if "success" in result:
                    success = bool(result["success"])
                    error_message = None if success else result.get("error")
                elif result.get("error"):
                    success = False
                    error_message = str(result["error"])
                else:
                    success = True
                    error_message = None
            else:
                success = True
                error_message = None
            return ToolCallResult(
                tool_name=tool_name, success=success, result=result,
                error=error_message, execution_time=execution_time,
                parameters=params,
            )
        except TypeError as error:
            try:
                sig = inspect.signature(tool_func)
                accepted = list(sig.parameters.keys())
            except Exception:
                accepted = []
            hint = (
                f"\n\n⚠️ 参数名错误。工具 `{tool_name}` 接受的关键字："
                f"{accepted}\n你传入的：{list(params.keys())}"
            )
            return ToolCallResult(
                tool_name=tool_name, success=False,
                error=str(error) + hint,
                execution_time=time.time() - start_time,
                parameters=params,
            )
        except Exception as error:
            return ToolCallResult(
                tool_name=tool_name, success=False, error=str(error),
                execution_time=time.time() - start_time,
                parameters=params,
            )
    def format_tool_result(self, result: ToolCallResult) -> str:
        from .format import format_tool_result as _fmt
        try:
            available = list(self.registry.get_all_metadata().keys())
        except Exception:
            available = None
        return _fmt(result, available_tools=available)

    def generate_confirmation_message(self, tool_call: ToolCallRequest) -> str:
        tool_meta = self.registry.get_metadata(tool_call.tool_name)
        tool_desc = tool_meta.description if tool_meta else ""
        message = f"🔧 即将执行工具: [{tool_call.tool_name}]\n"
        message += f"📋 说明: {tool_desc}\n"
        message += "⚠️ 此操作需要确认\n\n📥 参数:\n"
        for key, value in tool_call.parameters.items():
            message += f"  - {key}: {value}\n"
        message += "\n是否继续执行？回复\"确认\"或\"取消\""
        return message

    def get_tools_summary(self) -> str:
        return self.registry.generate_tools_prompt()


def create_agent() -> ToolAgent:
    return ToolAgent()