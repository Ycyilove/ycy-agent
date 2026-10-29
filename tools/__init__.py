"""
工具系统模块
提供装饰器模式统一管理所有可用工具，自动生成工具描述清单

使用示例:
    from tools import tool, get_registry

    @tool(name="my_tool", description="我的工具", required=["param"])
    def my_tool(param: str) -> str:
        return param
"""

import json
import re
import inspect
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from functools import wraps


# ── Python 类型名 → JSON Schema 类型名 ──
_PY_TYPE_MAP = {
    "str": "string", "string": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array",
    "dict": "object", "object": "object",
}


def _normalize_type(t: Any) -> str:
    """把 'str' / "<class 'str'>" / 'typing.List[str]' 等归一成 JSON Schema 类型名。"""
    if t is None:
        return "string"
    s = str(t).strip().lower()
    if "class '" in s:
        s = s.split("class '")[-1].split("'")[0].strip()
    if s.startswith("typing."):
        s = s[len("typing."):]
    if s.startswith("list["):
        return "array"
    if s.startswith("dict["):
        return "object"
    if s.startswith("optional["):
        inner = s[len("optional["):-1]
        return _normalize_type(inner)
    return _PY_TYPE_MAP.get(s, s if s in _PY_TYPE_MAP.values() else "string")


def _normalize_property(pinfo: Any) -> Dict[str, Any]:
    """把单个字段定义归一成标准 JSON Schema property。"""
    if not isinstance(pinfo, dict):
        return {"type": "string"}
    out = dict(pinfo)
    out["type"] = _normalize_type(out.get("type"))
    return out


def _build_input_schema(
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """把扁平 parameters 规范化为标准 JSON Schema。

    - 扁平形态 {field: {...}} → {"type":"object","properties":{...},"required":[...]}
    - 已是标准形态 {"properties":{...}} → 原样返回（幂等）
    """
    if not parameters:
        return None
    if "properties" in parameters and "type" in parameters:
        # 已是标准 JSON Schema，幂等返回
        return parameters
    props = {k: _normalize_property(v) for k, v in parameters.items()}
    return {
        "type": "object",
        "properties": props,
        "required": list(required or []),
    }


@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    parameters: Dict[str, Any]
    examples: List[str] = field(default_factory=list)
    category: str = "general"
    danger_level: str = "safe"
    input_schema: Optional[Dict[str, Any]] = None
    origin: str = "builtin"
    idempotent: bool = False
    resource_kinds: List[str] = field(default_factory=list)
    trust_level: int = 0                      # ← 新增：0 内置 / 1 静态 MCP / 2 动态 / 3 拒绝


class ToolRegistry:
    """工具注册中心"""

    _instance = None
    _tools: Dict[str, Callable] = {}
    _metadata: Dict[str, ToolMetadata] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tools = {}
            cls._metadata = {}
        return cls._instance

    def register(self, name: str = None, description: str = "",
                 parameters: Dict = None, examples: List[str] = None,
                 category: str = "general", danger_level: str = "safe",
                 resource_kinds: List[str] = None,
                 required: List[str] = None):
        """工具注册装饰器（支持 required 显式声明）"""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or ""

            tool_params = dict(parameters or {})   # 拷一份，避免副作用
            inferred_required: List[str] = []

            if not tool_params and func.__annotations__:
                try:
                    sig = inspect.signature(func)
                except (ValueError, TypeError):
                    sig = None
                for param_name, param_type in func.__annotations__.items():
                    if param_name == 'return':
                        continue
                    tool_params[param_name] = {
                        "type": _normalize_type(param_type),
                        "description": f"参数 {param_name}",
                    }
                    if sig is not None:
                        p = sig.parameters.get(param_name)
                        if p is not None and p.default is inspect.Parameter.empty:
                            inferred_required.append(param_name)

            # required 优先级：显式传入 > 注解推断
            final_required = list(required) if required is not None else inferred_required

            # 产出标准 input_schema（幂等）
            input_schema = _build_input_schema(tool_params, final_required)

            self.register_callable(
                func,
                name=tool_name,
                description=tool_description,
                parameters=tool_params,      # 保留扁平形态，向后兼容
                examples=examples,
                category=category,
                danger_level=danger_level,
                input_schema=input_schema,   # 新增：标准 JSON Schema
                resource_kinds=resource_kinds,
            )

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper.tool_name = tool_name
            wrapper.tool_metadata = self._metadata[tool_name]
            return wrapper
        return decorator

    def register_callable(
        self,
        func: Callable,
        name: str,
        description: str = "",
        parameters: Dict = None,
        examples: List[str] = None,
        category: str = "general",
        danger_level: str = "safe",
        input_schema: Dict[str, Any] = None,
        origin: str = "builtin",
        idempotent: bool = False,
        resource_kinds: List[str] = None,
        required: List[str] = None,
        trust_level: int = 0,                 # ← 新增
    ) -> Callable:
        if input_schema is None and parameters:
            input_schema = _build_input_schema(parameters, required)

        self._tools[name] = func
        self._metadata[name] = ToolMetadata(
            name=name,
            description=description or getattr(func, "__doc__", "") or "",
            parameters=dict(parameters or {}),
            examples=examples or [],
            category=category,
            danger_level=danger_level,
            input_schema=input_schema,
            origin=origin,
            idempotent=idempotent,
            resource_kinds=list(resource_kinds or []),
            trust_level=trust_level,
        )
        return func
        
    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def unregister(self, name: str) -> bool:
        existed = name in self._tools or name in self._metadata
        self._tools.pop(name, None)
        self._metadata.pop(name, None)
        return existed

    def get_all_tools(self) -> Dict[str, Callable]:
        return self._tools.copy()

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        return self._metadata.get(name)

    def get_all_metadata(self) -> Dict[str, ToolMetadata]:
        return self._metadata.copy()

    def list_tools(self, category: str = None) -> List[str]:
        if category:
            return [name for name, meta in self._metadata.items() if meta.category == category]
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> List[Dict]:
        descriptions = []
        for name, meta in self._metadata.items():
            descriptions.append({
                "name": name,
                "description": meta.description,
                "parameters": meta.parameters,
                "category": meta.category,
                "danger_level": meta.danger_level,
                "origin": meta.origin,
            })
        return descriptions

    def generate_tools_prompt(self) -> str:
        prompt_parts = ["可用工具：\n"]
        for name, meta in self._metadata.items():
            prompt_parts.append(f"\n## {name} ({meta.category})")
            prompt_parts.append(f"描述：{meta.description}")
            prompt_parts.append(f"危险等级：{meta.danger_level}")
            if meta.parameters:
                prompt_parts.append("参数：")
                for param, info in meta.parameters.items():
                    param_type = info.get("type", "any")
                    param_desc = info.get("description", "")
                    prompt_parts.append(f"  - {param} ({param_type}): {param_desc}")
            if meta.examples:
                prompt_parts.append("示例：")
                for ex in meta.examples:
                    prompt_parts.append(f"  {ex}")
        return "\n".join(prompt_parts)

    def clear(self):
        self._tools.clear()
        self._metadata.clear()


registry = ToolRegistry()


def tool(name: str = None, description: str = "",
         parameters: Dict = None, examples: List[str] = None,
         category: str = "general", danger_level: str = "safe",
         resource_kinds: List[str] = None,
         required: List[str] = None):
    """工具注册装饰器（简写）

    required: 显式声明必填参数名列表。为空时用函数注解推断（无默认值者）。
    """
    return registry.register(
        name=name,
        description=description,
        parameters=parameters,
        examples=examples,
        category=category,
        danger_level=danger_level,
        resource_kinds=resource_kinds,
        required=required,
    )


def get_registry() -> ToolRegistry:
    return registry