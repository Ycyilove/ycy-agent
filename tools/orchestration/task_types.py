"""re-export 壳（B11-1）。

实际定义已搬到 core/task/task_types.py。
本文件保持旧 import 不破（tools.orchestration.task_types.xxx）。
"""
from core.task.task_types import (
    TaskType,
    GapKind,
    StopReason,
    TERMINAL_REASONS,
    AssessResult,
    can_upgrade,
)

__all__ = [
    "TaskType",
    "GapKind",
    "StopReason",
    "TERMINAL_REASONS",
    "AssessResult",
    "can_upgrade",
]