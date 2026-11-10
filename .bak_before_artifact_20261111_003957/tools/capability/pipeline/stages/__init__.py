"""DiscoveryPipeline 的阶段。每个阶段是独立类。"""
from .normalize import NormalizeStage
from .cover import CoverStage
from .local_search import LocalSearchStage
from .trust import TrustStage

__all__ = [
    "NormalizeStage",
    "CoverStage",
    "LocalSearchStage",
    "TrustStage",
]