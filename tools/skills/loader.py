"""Skill 加载器：扫描 data/skills/ 并注册进 ToolRegistry。

每个 skill 目录:
    data/skills/<skill_id>/
        skill.json     # SkillDefinition 字段
        (可选) skill.py / prompt.md
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools import get_registry, ToolMetadata
from tools.skills.types import SkillDefinition

logger = logging.getLogger(__name__)

SKILLS_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "skills"


def _load_one(skill_dir: Path) -> Optional[SkillDefinition]:
    cfg = skill_dir / "skill.json"
    if not cfg.exists():
        return None
    try:
        raw = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("[skills] 解析失败: %s", cfg)
        return None
    raw.setdefault("id", skill_dir.name)
    raw.setdefault("name", raw["id"])
    raw.setdefault("description", "")
    raw.setdefault("tool_names", [])
    raw.setdefault("parameters", {})
    raw.setdefault("examples", [])
    try:
        return SkillDefinition(**raw)
    except TypeError:
        # 容错：只取已知字段
        known = {"id","name","description","prompt_template","tool_names",
                 "parameters","examples","category","danger_level",
                 "enabled","origin","version"}
        return SkillDefinition(**{k: v for k, v in raw.items() if k in known})


def list_skill_definitions() -> List[SkillDefinition]:
    if not SKILLS_ROOT.exists():
        return []
    out: List[SkillDefinition] = []
    for d in sorted(SKILLS_ROOT.iterdir()):
        if d.is_dir():
            s = _load_one(d)
            if s:
                out.append(s)
    return out


def get_skill(skill_id: str) -> Optional[SkillDefinition]:
    for s in list_skill_definitions():
        if s.id == skill_id:
            return s
    return None


def save_skill(skill: SkillDefinition) -> None:
    d = SKILLS_ROOT / skill.id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": skill.id, "name": skill.name, "description": skill.description,
        "prompt_template": skill.prompt_template,
        "tool_names": skill.tool_names, "parameters": skill.parameters,
        "examples": skill.examples, "category": skill.category,
        "danger_level": skill.danger_level, "enabled": skill.enabled,
        "version": skill.version,
    }
    (d / "skill.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def delete_skill(skill_id: str) -> bool:
    d = SKILLS_ROOT / skill_id
    if not d.exists():
        return False
    import shutil
    shutil.rmtree(d)
    return True


def register_skills_into_registry() -> List[str]:
    """把启用中的 skill 注册为 origin='skill' 的工具元数据（只注册元数据占位）。

    ⚠️ 安全：不再 exec skill.py。Skill 是「工具组合 + prompt」，
    执行落到 tool_names 里的工具链。
    """
    reg = get_registry()
    registered: List[str] = []
    for skill in list_skill_definitions():
        if not skill.enabled:
            continue

        def handler(**kwargs):  # noqa: E306
            return {
                "success": True,
                "skill": skill.id,
                "note": "skill 由 orchestrator 注入 prompt，工具链见 tool_names",
                "tool_names": skill.tool_names,
            }

        reg.register_callable(
            handler,
            name=skill.id,
            description=skill.description,
            parameters=skill.parameters,
            examples=skill.examples,
            category=skill.category,
            danger_level=skill.danger_level,
            origin="skill",
        )
        registered.append(skill.id)
    logger.info("[skills] registered %d skills", len(registered))
    return registered


def reload_skills() -> List[str]:
    reg = get_registry()
    # 先移除旧 skill 元数据
    for name, meta in list(reg.get_all_metadata().items()):
        if meta.origin == "skill":
            reg.unregister(name)
    return register_skills_into_registry()