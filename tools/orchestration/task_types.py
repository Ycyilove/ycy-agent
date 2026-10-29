# tools/orchestration/task_types.py  完整内容（新建）
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class TaskType(str, Enum):
    CHITCHAT = "chitchat"
    INFO_QUERY = "info_query"
    WEB_RESEARCH = "web_research"
    DOC_SURVEY = "doc_survey"
    FILE_OP = "file_op"
    CODE_FIX = "code_fix"
    GENERATE = "generate"


_PRIORITY = {
    TaskType.CHITCHAT: 0,
    TaskType.INFO_QUERY: 1,
    TaskType.WEB_RESEARCH: 2,
    TaskType.DOC_SURVEY: 3,
    TaskType.CODE_FIX: 4,
    TaskType.GENERATE: 5,
    TaskType.FILE_OP: 6,
}


def can_upgrade(current: TaskType, target: TaskType) -> bool:
    if current == TaskType.CODE_FIX and target != TaskType.CODE_FIX:
        return False
    return _PRIORITY.get(target, 0) > _PRIORITY.get(current, 0)


class StopReason(str, Enum):
    DONE = "done"
    MAX_STEPS = "max_steps"
    NEEDS_USER = "needs_user"
    BLOCKED = "blocked"
    PARTIAL = "partial"
    QUALITY_FAIL = "quality_fail"
    ABORTED = "aborted"


TERMINAL_REASONS = frozenset({
    StopReason.DONE, StopReason.MAX_STEPS, StopReason.NEEDS_USER,
    StopReason.BLOCKED, StopReason.PARTIAL, StopReason.QUALITY_FAIL,
    StopReason.ABORTED,
})


@dataclass
class AssessResult:
    sufficient: bool
    stop_reason: StopReason = StopReason.DONE
    gap: str = ""
    next_hint: str = ""
    verify_detail: Optional[dict] = field(default=None)

    @property
    def should_terminate(self) -> bool:
        if self.sufficient:
            return True
        return self.stop_reason in TERMINAL_REASONS and self.stop_reason != StopReason.NEEDS_USER

    @property
    def should_continue(self) -> bool:
        return not self.sufficient and self.stop_reason == StopReason.NEEDS_USER

    def to_dict(self) -> dict:
        return {
            "sufficient": self.sufficient,
            "stop_reason": self.stop_reason.value,
            "gap": self.gap,
            "next_hint": self.next_hint,
        }