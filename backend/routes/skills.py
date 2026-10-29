"""技能（Skills）管理路由。"""
from __future__ import annotations
import json
import traceback
from typing import Any, Callable, List, Optional

from fastapi import APIRouter, Body, HTTPException

from tools.skills.loader import (
    list_skill_definitions, get_skill, save_skill, delete_skill,
    reload_skills,
)
from tools.skills.types import SkillDefinition
from tools.skills.session_skills import (
    get_enabled_skill_ids, set_enabled_skill_ids,
)

def create_skills_router(agent_provider: Callable[[], Any]) -> APIRouter:
    router = APIRouter()

    @router.get("/api/skills")
    async def list_skills():
        try:
            items = [s.to_metadata() for s in list_skill_definitions()]
            return {"status": "success", "skills": items, "count": len(items)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/api/skills/{skill_id}")
    async def get_skill_info(skill_id: str):
        skill = get_skill(skill_id)
        if not skill:
            raise HTTPException(status_code=404, detail=f"技能不存在: {skill_id}")
        return {"status": "success", "skill": skill.to_metadata()}

    @router.post("/api/skills")
    async def create_skill(payload: dict = Body(...)):
        try:
            sid = (payload.get("id") or "").strip()
            if not sid:
                raise HTTPException(status_code=400, detail="id 必填")
            if get_skill(sid):
                raise HTTPException(status_code=409, detail=f"技能已存在: {sid}")
            skill = SkillDefinition(
                id=sid,
                name=payload.get("name", sid),
                description=payload.get("description", ""),
                prompt_template=payload.get("prompt_template", ""),
                tool_names=payload.get("tool_names", []) or [],
                parameters=payload.get("parameters", {}) or {},
                examples=payload.get("examples", []) or [],
                category=payload.get("category", "skill"),
                danger_level=payload.get("danger_level", "safe"),
                enabled=bool(payload.get("enabled", True)),
                version=payload.get("version", "1.0"),
            )
            save_skill(skill)
            reload_skills()
            return {"status": "success", "skill": skill.to_metadata()}
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/api/skills/{skill_id}")
    async def update_skill(skill_id: str, payload: dict = Body(...)):
        try:
            existing = get_skill(skill_id)
            if not existing:
                raise HTTPException(status_code=404, detail=f"技能不存在: {skill_id}")
            merged = existing.to_metadata()
            merged.update({k: v for k, v in payload.items() if k != "id"})
            merged["id"] = skill_id
            skill = SkillDefinition(
                id=skill_id,
                name=merged.get("name", existing.name),
                description=merged.get("description", existing.description),
                prompt_template=merged.get("prompt_template", existing.prompt_template),
                tool_names=merged.get("tool_names", existing.tool_names),
                parameters=merged.get("parameters", existing.parameters),
                examples=merged.get("examples", existing.examples),
                category=merged.get("category", existing.category),
                danger_level=merged.get("danger_level", existing.danger_level),
                enabled=bool(merged.get("enabled", existing.enabled)),
                version=merged.get("version", existing.version),
            )
            save_skill(skill)
            reload_skills()
            return {"status": "success", "skill": skill.to_metadata()}
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/api/skills/{skill_id}")
    async def remove_skill(skill_id: str):
        if not delete_skill(skill_id):
            raise HTTPException(status_code=404, detail=f"技能不存在: {skill_id}")
        reload_skills()
        return {"status": "success", "deleted": skill_id}

    @router.post("/api/skills/{skill_id}/toggle")
    async def toggle_skill(skill_id: str, enabled: bool = Body(..., embed=True)):
        skill = get_skill(skill_id)
        if not skill:
            raise HTTPException(status_code=404, detail=f"技能不存在: {skill_id}")
        skill.enabled = enabled
        save_skill(skill)
        reload_skills()
        return {"status": "success", "skill": skill.to_metadata()}

    @router.post("/api/skills/reload")
    async def reload_all():
        names = reload_skills()
        return {"status": "success", "registered": names}

    # ── 会话级 Skills 启用状态 ──
    @router.get("/api/sessions/{session_id}/skills")
    async def get_session_skills(session_id: str):
        return {
            "status": "success",
            "session_id": session_id,
            "enabled_skills": get_enabled_skill_ids(session_id),
        }

    @router.put("/api/sessions/{session_id}/skills")
    async def put_session_skills(session_id: str, payload: dict = Body(...)):
        ids = payload.get("enabled_skills") or []
        set_enabled_skill_ids(session_id, list(ids))
        return {
            "status": "success",
            "session_id": session_id,
            "enabled_skills": ids,
        }

    return router