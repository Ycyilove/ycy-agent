"""会话级 Skills 启用状态。

存储：data/skills_sessions/<session_id>.json
    {"enabled_skills": ["skill_a", "skill_b"], "updated_at": "..."}

职责：
- 前端 SkillsPanel 勾选"本会话启用哪些 skill" → 写这里
- Orchestrator.run() 开头读这里 → 解析成工具名集合 → set_available_tools
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from tools.skills.loader import get_skill, list_skill_definitions

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "skills_sessions"


def _path(session_id: str) -> Path:
    safe = "".join(c for c in (session_id or "") if c.isalnum() or c in "-_")
    return SESSIONS_DIR / f"{safe or '_default'}.json"


def get_enabled_skill_ids(session_id: Optional[str]) -> List[str]:
    if not session_id:
        return []
    p = _path(session_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return list(data.get("enabled_skills") or [])
    except Exception:
        logger.exception("[skills] 读取会话启用状态失败: %s", p)
        return []


def set_enabled_skill_ids(session_id: str, skill_ids: List[str]) -> None:
    # ── 校验 skill 真实存在，防注入无效 id ──
    valid_ids = {s.id for s in list_skill_definitions()}
    filtered = [sid for sid in (skill_ids or []) if sid in valid_ids]
    if len(filtered) != len(skill_ids or []):
        logger.warning(
            "[skills] set_enabled_skill_ids: 过滤无效 skill id, 原始=%r, 有效=%r",
            skill_ids, filtered,
        )

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(session_id)
    p.write_text(
        json.dumps(
            {"enabled_skills": filtered, "updated_at": time.time()},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


def build_skills_prompt_block(session_id: Optional[str]) -> str:
    """把本会话启用的 skills 渲染成注入 prompt 的文本块。"""
    enabled = get_enabled_skill_ids(session_id)
    if not enabled:
        return ""
    lines: List[str] = ["## 本会话启用的技能（Skills）"]
    for sid in enabled:
        skill = get_skill(sid)
        if skill is None or not skill.enabled:
            continue
        lines.append(f"\n### 技能 `{skill.id}`：{skill.name}")
        if skill.description:
            lines.append(skill.description)
        if skill.prompt_template:
            lines.append(skill.prompt_template)
        if skill.tool_names:
            lines.append(f"建议工具链：{', '.join(skill.tool_names)}")
    if len(lines) == 1:
        return ""
    lines.append("\n**注意**：技能是「建议路径」，不是硬约束；若明显偏离用户目标，按通用策略处理。\n")
    return "\n".join(lines)