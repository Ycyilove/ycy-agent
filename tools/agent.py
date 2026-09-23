"""ToolAgent + IntentRecognizer.

- 引导提取（guidance）已移至 tools/guidance/
- 幂等分类（idempotency）已移至 tools/idempotency/
- 失败分类（failure）已移至 tools/semantics/
- 本模块只保留 ToolAgent 主逻辑 + 规则/LLM 意图识别
- 兼容旧 import：re-export 上述函数
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import registry, get_registry

# 兼容 re-export
from .guidance import extract_guidance
from .idempotency import classify_tool_idempotency
from .semantics import (
    classify_failure,
    get_failure_recovery_hint,
)


logger = logging.getLogger(__name__)


_MCP_TOOL_PREFIX = "mcp__"


# ─────────────────────────────────────────────────────────────
# 意图识别
# ─────────────────────────────────────────────────────────────

LLM_INTENT_PROMPT = """你是一个智能工具选择助手。根据用户消息和可用工具列表，选择最合适的工具并提取参数。

## 可用工具列表
{tool_descriptions}

## 用户消息
{user_message}

## 输出格式（严格 JSON）
{{
    "needs_tool": true或false,
    "tool_name": "工具名",
    "parameters": {{}},
    "reasoning": "理由"
}}

## 关键规则
1. 只选择明确匹配的工具。
2. 闲聊/问候不触发工具。
3. 只使用列表中的工具。
4. 避免过度解读。"""


class LLMIntentRecognizer:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._tool_descriptions_cache = None

    def _get_llm_client(self):
        if self.llm_client:
            return self.llm_client
        try:
            from langchain_service import get_llm
            return get_llm()
        except ImportError:
            return None

    def _get_tool_descriptions(self, registry_instance) -> str:
        if self._tool_descriptions_cache is None:
            metadata_dict = registry_instance.get_all_metadata()
            descriptions = []
            for tool_name, tool in metadata_dict.items():
                schema = getattr(tool, "input_schema", None) or {}
                required = set(schema.get("required") or [])
                properties = schema.get("properties") or tool.parameters or {}
                param_parts = []
                for name, info in properties.items():
                    if not isinstance(info, dict):
                        param_parts.append(f"{name}: any")
                        continue
                    ptype = info.get("type", "str")
                    mark = "必填" if name in required else "可选"
                    param_parts.append(f"{name}: {ptype} ({mark})")
                param_str = ", ".join(param_parts) if param_parts else "无"
                descriptions.append(f"- {tool.name}: {tool.description} (参数: {param_str})")
            self._tool_descriptions_cache = "\n".join(descriptions)
        return self._tool_descriptions_cache

    def analyze(self, user_message: str, registry_instance=None) -> "IntentAnalysis":
        if registry_instance is None:
            registry_instance = get_registry()
        llm = self._get_llm_client()
        if not llm:
            return IntentAnalysis(
                needs_tool=False, intent_type="llm_unavailable",
                confidence=0.0, reasoning="LLM不可用",
            )

        tool_descriptions = self._get_tool_descriptions(registry_instance)
        prompt = LLM_INTENT_PROMPT.format(
            tool_descriptions=tool_descriptions,
            user_message=user_message,
        )
        try:
            response = llm.generate([{"role": "user", "content": prompt}])
            clean = response.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
            result = json.loads(clean)
            tool_name = result.get("tool_name")
            return IntentAnalysis(
                needs_tool=result.get("needs_tool", False),
                intent_type=f"llm_{tool_name or 'conversation'}",
                confidence=0.95,
                suggested_tool=tool_name,
                parameters=result.get("parameters") or {},
                reasoning=result.get("reasoning", ""),
            )
        except json.JSONDecodeError as e:
            return IntentAnalysis(
                needs_tool=False, intent_type="parse_error",
                confidence=0.0, reasoning=f"LLM返回格式错误: {e}",
            )
        except Exception as e:
            return IntentAnalysis(
                needs_tool=False, intent_type="error",
                confidence=0.0, reasoning=f"LLM分析失败: {e}",
            )


# ─────────────────────────────────────────────────────────────
# 规则意图识别（保留完整版，未拆分）
# ─────────────────────────────────────────────────────────────

class IntentRecognizer:
    """规则意图识别器（关键词匹配）。"""

    def __init__(self):
        self.intent_patterns = {
            "export_to_csv": {
                "keywords": ["导出", "导出csv", "保存csv", "输出csv", "export csv"],
                "tool": "export_to_csv",
                "priority": 10,
                "param_extractors": {
                    "data": self._extract_json_data,
                    "file_path": self._extract_export_file_path,
                    "columns": self._extract_columns,
                },
            },
            "run_code": {
                "keywords": ["运行代码", "执行代码", "run code", "python", "计算", "算一下"],
                "tool": "run_python_code",
                "priority": 5,
                "param_extractors": {"code": self._extract_code_block},
            },
            "read_csv": {
                "keywords": ["读取csv", "读csv", "打开csv", "read csv"],
                "tool": "read_csv",
                "priority": 5,
                "param_extractors": {"file_path": self._extract_file_path},
            },
            "list_files": {
                "keywords": ["列出文件", "查看文件", "目录", "list files", "ls", "dir"],
                "tool": "list_files",
                "priority": 5,
                "param_extractors": {"directory": self._extract_directory},
            },
            "check_url": {
                "keywords": ["检查网址", "网址状态", "检查网站", "检查链接"],
                "tool": "check_url_status",
                "priority": 6,
                "param_extractors": {"url": self._extract_url},
            },
            "fetch_json": {
                "keywords": ["获取json", "api数据", "fetch json"],
                "tool": "fetch_json",
                "param_extractors": {"url": self._extract_url},
            },
            "calculate_age": {
                "keywords": ["年龄", "几岁", "多大", "age", "岁数"],
                "tool": "calculate_age",
                "priority": 5,
                "param_extractors": {"birth_date": self._extract_birth_date},
            },
        }
        self._nlp_processor = None
        try:
            from .nlp_processor import get_nlp_processor
            self._nlp_processor = get_nlp_processor()
        except ImportError:
            pass

    # 参数提取器（保留核心几个，其余可按需补）
    def _extract_code_block(self, text: str) -> Optional[str]:
        patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'计算\s*([\d\s\+\-\*\/\(\)]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _extract_file_path(self, text: str) -> Optional[str]:
        patterns = [
            r'["\']([^"\']+\.(csv|xlsx?|txt|json|xml))["\']',
            r'(?:文件|path)[:：]\s*([^\s]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_directory(self, text: str) -> Optional[str]:
        patterns = [
            r'(?:目录|文件夹|directory)[:：]\s*([^\s]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return "."

    def _extract_url(self, text: str) -> Optional[str]:
        match = re.search(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
        return match.group(0) if match else None

    def _extract_json_data(self, text: str) -> Optional[List[Dict]]:
        match = re.search(r'(\[.*?\])', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return None

    def _extract_export_file_path(self, text: str) -> Optional[str]:
        match = re.search(r'(?:到|为)\s+([^\s]+\.csv)', text)
        return match.group(1).strip() if match else None

    def _extract_columns(self, text: str) -> Optional[List[str]]:
        match = re.search(r'列\s*[:：]\s*\[(.*?)\]', text)
        if match:
            return [c.strip().strip('"\'') for c in match.group(1).split(',')]
        return None

    def _extract_birth_date(self, text: str) -> Optional[str]:
        match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', text)
        return match.group(1) if match else None

    def analyze(self, user_message: str) -> "IntentAnalysis":
        nlp_info = {}
        if self._nlp_processor:
            try:
                nlp_info = self._nlp_processor.preprocess_for_intent(user_message)
            except Exception:
                pass

        message_lower = user_message.lower()
        tokens_lower = [t.lower() for t in nlp_info.get("tokens", [])]

        best_match = None
        best_score = 0.0
        best_keyword = None

        for intent_name, cfg in sorted(
            self.intent_patterns.items(),
            key=lambda x: x[1].get("priority", 0),
            reverse=True,
        ):
            matched = []
            score = 0.0
            for keyword in cfg["keywords"]:
                kw = keyword.lower()
                if kw in message_lower:
                    matched.append(keyword)
                    score += 1.0 + len(keyword) * 0.1
                if tokens_lower and kw in tokens_lower:
                    if keyword not in matched:
                        matched.append(keyword)
                    score += 1.5
            if matched and score > best_score:
                best_match = intent_name
                best_score = score
                best_keyword = matched[0]

        if best_match and best_keyword:
            cfg = self.intent_patterns[best_match]
            parameters = {}
            for pname, extractor in cfg["param_extractors"].items():
                try:
                    v = extractor(user_message)
                    if v:
                        parameters[pname] = v
                except Exception:
                    pass
            confidence = min(0.95, 0.5 + best_score * 0.05)
            return IntentAnalysis(
                needs_tool=True,
                intent_type=best_match,
                confidence=confidence,
                suggested_tool=cfg["tool"],
                parameters=parameters,
                reasoning=f"检测到关键词: {best_keyword}",
            )

        return IntentAnalysis(
            needs_tool=False,
            intent_type="general_conversation",
            confidence=0.0,
        )


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


@dataclass
class IntentAnalysis:
    needs_tool: bool
    intent_type: str
    confidence: float
    suggested_tool: str = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class ToolAgent:
    """工具调用决策和执行。"""

    def __init__(self, registry: registry = None):
        self.registry = registry or get_registry()
        self.rule_intent_recognizer = IntentRecognizer()
        self.llm_intent_recognizer = LLMIntentRecognizer()
        self.tool_descriptions = self.registry.get_tool_descriptions()
        self._mcp_client_manager = None
        self._allowed_tools: Optional[set] = None
        
    def set_mcp_client_manager(self, manager) -> None:
        self._mcp_client_manager = manager
        logger.info(
            "[dsh][mcp] ToolAgent bound to MCPClientManager: %r",
            type(manager).__name__,
        )

    # [新增] 设置本次会话允许使用的工具集合
    def set_available_tools(self, tool_names: List[str]) -> None:
        if not tool_names:
            self._allowed_tools = None
        else:
            self._allowed_tools = set(tool_names)
        logger.info(f"[Skills] Allowed tools set to: {len(tool_names)} tools")

    @staticmethod
    def _is_mcp_tool_name(tool_name: str) -> bool:
        return (
            isinstance(tool_name, str)
            and tool_name.startswith(_MCP_TOOL_PREFIX)
            and "__" in tool_name[len(_MCP_TOOL_PREFIX):]
        )

    def register_dynamic_tool(self, name, description, parameters=None,
                               handler=None, input_schema=None, origin="builtin",
                               danger_level="safe", idempotent=False):
        logger.info("[dsh][mcp] register tool: %s (origin=%s)", name, origin)
        self.registry.register_callable(
            handler, name=name, description=description,
            parameters=parameters, input_schema=input_schema,
            origin=origin, danger_level=danger_level, idempotent=idempotent,
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
            # [修改] 增加技能过滤逻辑
            if self._allowed_tools is not None and name not in self._allowed_tools:
                continue
            if allowed_danger_levels is not None and meta.danger_level not in allowed_danger_levels:
                continue
            result[name] = meta
        return result

    def analyze_intent(self, user_message: str, use_llm: bool = True) -> IntentAnalysis:
        if use_llm:
            llm_result = self.llm_intent_recognizer.analyze(user_message, self.registry)
            # [修改] 如果识别出的工具不在允许列表中，则视为未识别
            if llm_result.suggested_tool:
                if self._allowed_tools is None or llm_result.suggested_tool in self._allowed_tools:
                    return llm_result
        rule_result = self.rule_intent_recognizer.analyze(user_message)
        if rule_result.suggested_tool:
             # [修改] 规则识别也要受技能限制
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

        # 动态 MCP 路由
        if tool_func is None and self._is_mcp_tool_name(tool_name):
            mcp_client = self._mcp_client_manager
            if mcp_client is None:
                return ToolCallResult(
                    tool_name=tool_name, success=False,
                    error=f"工具 {tool_name} 未注册且 MCPClientManager 未注入。",
                    execution_time=time.time() - start_time,
                )
            logger.info("[dsh][mcp] dynamic route: %s", tool_name)
            try:
                result = await mcp_client.call_tool(tool_name, tool_call.parameters or {})
                if isinstance(result, dict):
                    success = result.get("success", True)
                    error_message = None if success else result.get("error")
                else:
                    success = True
                    error_message = None
                return ToolCallResult(
                    tool_name=tool_name, success=success, result=result,
                    error=error_message, execution_time=time.time() - start_time,
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
                )

        if tool_func is None:
            return ToolCallResult(
                tool_name=tool_name, success=False,
                error=f"工具不存在: {tool_name}",
                execution_time=time.time() - start_time,
            )

        try:
            result = tool_func(**tool_call.parameters)
            if inspect.isawaitable(result):
                result = await result
            execution_time = time.time() - start_time
            if isinstance(result, dict):
                success = result.get("success", True)
                error_message = None if success else result.get("error")
            else:
                success = True
                error_message = None
            return ToolCallResult(
                tool_name=tool_name, success=success, result=result,
                error=error_message, execution_time=execution_time,
            )
        except Exception as error:
            return ToolCallResult(
                tool_name=tool_name, success=False, error=str(error),
                execution_time=time.time() - start_time,
            )

    def format_tool_result(self, result: ToolCallResult) -> str:
        """格式化工具结果。

        - 用 guidance.extractor 提取引导
        - 用 semantics.failure 分类失败
        - 按 guidance_type 分级输出
        """
        guidance: Dict[str, Any] = {}
        if isinstance(result.result, dict):
            try:
                guidance = extract_guidance(result.result)
            except Exception:
                logger.exception("extract_guidance failed")

        next_tool = guidance.get("next_tool")
        next_args = guidance.get("next_args") or {}
        next_message = guidance.get("message") or ""
        guest_tools = guidance.get("guest_tools") or []
        guidance_type = guidance.get("guidance_type", "none")

        # 补 mcp__server__ 前缀
        if next_tool and not next_tool.startswith("mcp__") and "__" not in next_tool:
            current = result.tool_name
            if current.startswith("mcp__"):
                parts = current.split("__", 2)
                if len(parts) == 3:
                    next_tool = f"mcp__{parts[1]}__{next_tool}"

        directive_block = ""
        if next_tool and guidance_type == "mandatory":
            directive_block += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            directive_block += (
                "⛔ 任务未完成。必须先完成前置步骤：\n"
                if result.success
                else "❌ 工具调用被 MCP 服务器拒绝。\n"
            )
            directive_block += f"   1. 调用 {next_tool}\n"
            if next_args:
                directive_block += f"      参数: {json.dumps(next_args, ensure_ascii=False)}\n"
            if next_message:
                directive_block += f"      原因: {next_message[:300]}\n"
            directive_block += "   **在完成这一步之前，禁止调用其他工具。**\n"
            directive_block += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        elif next_tool and guidance_type == "optional":
            directive_block += "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈\n"
            directive_block += "💡 MCP 服务器附带了一个可选建议（与当前目标可能无关）：\n"
            directive_block += f"   - 可尝试 {next_tool}"
            if next_args:
                directive_block += f"，参数 {json.dumps(next_args, ensure_ascii=False)}"
            directive_block += "\n"
            if next_message:
                directive_block += f"   - 服务器说明: {next_message[:300]}\n"
            directive_block += (
                "   **这是可选建议。如果它与用户目标无关，请忽略，"
                "直接采取其他方式达成目标。**\n"
            )
            directive_block += "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈\n\n"

        # 成功分支
        if result.success:
            output = directive_block
            output += f"✅ 工具 [{result.tool_name}] 执行成功\n"
            output += f"⏱️ 耗时: {result.execution_time:.3f}s\n\n"

            if isinstance(result.result, dict):
                if "tools" in result.result and isinstance(result.result["tools"], list):
                    tools_list = result.result["tools"]
                    if tools_list:
                        output += "🔧 已连接 MCP 服务器。**下一步必须使用以下确切工具名**：\n"
                        for t in tools_list:
                            output += f"  - {t}\n"
                        output += "\n"
                if "message" in result.result:
                    output += f"📝 结果: {result.result['message']}\n"
                if "output" in result.result:
                    output += f"📤 输出:\n{result.result['output']}\n"
                if "data" in result.result:
                    output += (
                        f"📦 数据: "
                        f"{json.dumps(result.result['data'], ensure_ascii=False)[:500]}...\n"
                    )
                if guest_tools:
                    output += f"\n✅ 无需认证的工具：{', '.join(guest_tools)}\n"

                structured = result.result.get("structuredContent")
                if structured is not None:
                    try:
                        structured_json = json.dumps(structured, ensure_ascii=False, indent=2)
                        output += f"\n📊 结构化数据:\n{structured_json[:2000]}\n"
                    except Exception:
                        pass
            else:
                output += f"📤 结果: {result.result}\n"

            return output

        # 失败分支：用失败分类
        error_message = result.error or "(未知错误)"
        failure_type = classify_failure(error_message)

        # 从可用工具里取候选（如果 registry 有）
        available: List[str] = []
        try:
            available = list(self.registry.get_all_metadata().keys())
        except Exception:
            pass

        recovery_hint = get_failure_recovery_hint(
            failure_type=failure_type,
            error_message=error_message,
            failed_tool_name=result.tool_name,
            available_tools=available,
        )

        output = directive_block
        output += recovery_hint
        output += f"❌ 工具 [{result.tool_name}] 执行失败\n💥 错误: {error_message}\n"

        if isinstance(result.result, dict):
            content_blocks = result.result.get("content") or []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text") or ""
                    if text and text != error_message and len(text) < 800:
                        output += f"\n📄 详细错误:\n{text}\n"
                        break
            structured = result.result.get("structuredContent")
            if isinstance(structured, dict):
                try:
                    output += (
                        f"\n📊 结构化数据:\n"
                        f"{json.dumps(structured, ensure_ascii=False, indent=2)[:2000]}\n"
                    )
                except Exception:
                    pass

        return output

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