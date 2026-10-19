"""工具调用的意图识别（LLM + 规则两版）。"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from .types import IntentAnalysis

logger = logging.getLogger(__name__)


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
            from core.llm.provider import ModelScopeLLM
            return ModelScopeLLM()
        except ImportError:
            logger.warning(
                "[dsh][intent] core.llm.provider 不可用，LLM 意图识别禁用"
            )
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
