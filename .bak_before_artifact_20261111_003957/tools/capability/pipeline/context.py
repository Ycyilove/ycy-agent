"""DiscoveryContext：跨阶段传递状态。

字段：
    frame         IntentFrame（normalize 产出）
    manager       MCPClientManager
    register_tool / unregister_tool  工具注册回调
    allowed_danger_levels  安全等级白名单
    candidates    Registry 候选（registry 产出）
    tried         已尝试的 server（contextvar per-task）
    errors        连接错误汇总
    result        最终 discovery 结果（connect 产出）
    mode          "local"（maybe_discover） / "registry"（search_and_connect_mcp）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryContext:
    """一次 discovery 的完整上下文。"""

    mode: str = "local"
    manager: Any = None
    register_tool: Optional[Callable[..., Any]] = None
    unregister_tool: Optional[Callable[[str], Any]] = None
    allowed_danger_levels: Optional[set] = None

    # 阶段间传递
    frame: Any = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    tried: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None

    # 输入
    user_query: str = ""
    capability: str = ""
    limit: int = 100

    # 短路：cover 命中时设此标志
    short_circuit: bool = False

    # 观测
    stage_log: List[str] = field(default_factory=list)

    def log_stage(self, stage: str, msg: str) -> None:
        entry = f"[{stage}] {msg}"
        self.stage_log.append(entry)
        logger.info("[dsh][pipeline] %s", entry)