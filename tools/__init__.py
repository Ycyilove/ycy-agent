"""
工具系统模块
提供装饰器模式统一管理所有可用工具，自动生成工具描述清单

使用示例:
    from tools import tool, get_registry
    
    @tool(name="my_tool", description="我的工具")
    def my_tool(param: str) -> str:
        return param
    
    # 获取工具列表
    registry = get_registry()
    tools = registry.list_tools()
"""

import json
import re
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from functools import wraps


@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    parameters: Dict[str, Any]
    examples: List[str] = field(default_factory=list)
    category: str = "general"
    danger_level: str = "safe"


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

    def register(self, name: str = None, description: str = "", parameters: Dict = None,
                 examples: List[str] = None, category: str = "general", danger_level: str = "safe"):
        """工具注册装饰器"""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or ""

            tool_params = parameters or {}
            if not tool_params and func.__annotations__:
                for param_name, param_type in func.__annotations__.items():
                    if param_name != 'return':
                        tool_params[param_name] = {
                            "type": str(param_type).__replace("typing.", ""),
                            "description": f"参数 {param_name}"
                        }

            self._tools[tool_name] = func
            self._metadata[tool_name] = ToolMetadata(
                name=tool_name,
                description=tool_description,
                parameters=tool_params,
                examples=examples or [],
                category=category,
                danger_level=danger_level
            )

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper.tool_name = tool_name
            wrapper.tool_metadata = self._metadata[tool_name]

            return wrapper

        return decorator

    def get_tool(self, name: str) -> Optional[Callable]:
        """获取工具函数"""
        return self._tools.get(name)

    def get_all_tools(self) -> Dict[str, Callable]:
        """获取所有工具"""
        return self._tools.copy()

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """获取工具元数据"""
        return self._metadata.get(name)

    def get_all_metadata(self) -> Dict[str, ToolMetadata]:
        """获取所有工具元数据"""
        return self._metadata.copy()

    def list_tools(self, category: str = None) -> List[str]:
        """列出工具名称"""
        if category:
            return [name for name, meta in self._metadata.items() if meta.category == category]
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> List[Dict]:
        """生成工具描述清单（用于LLM）"""
        descriptions = []
        for name, meta in self._metadata.items():
            descriptions.append({
                "name": name,
                "description": meta.description,
                "parameters": meta.parameters,
                "category": meta.category,
                "danger_level": meta.danger_level
            })
        return descriptions

    def generate_tools_prompt(self) -> str:
        """生成工具说明提示词"""
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
        """清空注册表（主要用于测试）"""
        self._tools.clear()
        self._metadata.clear()


registry = ToolRegistry()


def tool(name: str = None, description: str = "", parameters: Dict = None,
         examples: List[str] = None, category: str = "general", danger_level: str = "safe"):
    """工具注册装饰器（简写）"""
    return registry.register(
        name=name,
        description=description,
        parameters=parameters,
        examples=examples,
        category=category,
        danger_level=danger_level
    )


def get_registry() -> ToolRegistry:
    """获取工具注册中心实例"""
    return registry
