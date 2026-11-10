"""Skill 元数据与定义文件 schema。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillDefinition:
    """一个 skill 的磁盘定义（data/skills/<id>/skill.json）。"""
    id: str
    name: str
    description: str
    prompt_template: str = ""            # 注入到 LLM 的技能说明/示例
    tool_names: List[str] = field(default_factory=list)  # 该 skill 启用的工具
    parameters: Dict[str, Any] = field(default_factory=dict)  # 暴露给用户填参
    examples: List[str] = field(default_factory=list)
    category: str = "skill"
    danger_level: str = "safe"
    enabled: bool = True
    origin: str = "skill"
    version: str = "1.0"

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.id,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
            "danger_level": self.danger_level,
            "origin": self.origin,
            "examples": self.examples,
            "enabled": self.enabled,
            "version": self.version,
            "tool_names": self.tool_names,
            "prompt_template": self.prompt_template,
        }