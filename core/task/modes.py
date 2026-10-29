"""Agent 工作模式：唯一真源。

ask   —— 对话模式：只读，单步直答，不进入编排循环。
agent —— 代理执行模式：全工具，多步编排，工作区内写操作免审批。

任何其他值都是非法输入，直接拒绝（不静默降级）。
"""

from __future__ import annotations

MODE_ASK = "ask"
MODE_AGENT = "agent"

MODES = frozenset({MODE_ASK, MODE_AGENT})

# 仅用于启动迁移，运行时不使用
LEGACY_MODES = frozenset({"plan", "craft", "auto"})


def validate_mode(mode: str) -> str:
    """校验模式。非法值抛 ValueError，不静默降级。"""
    if mode not in MODES:
        raise ValueError(f"非法 mode={mode!r}，必须是 {sorted(MODES)} 之一")
    return mode