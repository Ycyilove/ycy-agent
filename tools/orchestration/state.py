"""re-export 壳（B11-1）。

实际定义已搬到 core/task/state.py。
本文件保持旧 import 不破（tools.orchestration.state.xxx）。
"""
from core.task.state import OrchestratorState, params_key

__all__ = ["OrchestratorState", "params_key"]