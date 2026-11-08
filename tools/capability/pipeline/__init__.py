"""DiscoveryPipeline + 两条入口薄壳（B7-c）。

导出：
    - DiscoveryContext / DiscoveryPipeline（7 阶段编排）
    - maybe_discover（local 模式薄壳，原 tools/capability/pipeline.py）
    - search_and_connect_mcp（registry 模式薄壳，原 tools/mcp_discovery.py 的同名函数）
"""
from .context import DiscoveryContext
from .facade import maybe_discover, search_and_connect_mcp

__all__ = [
    "DiscoveryContext",
    "DiscoveryPipeline",
    "maybe_discover",
    "search_and_connect_mcp",
]


def __getattr__(name):
    """延迟 import DiscoveryPipeline。"""
    if name == "DiscoveryPipeline":
        from .pipeline import DiscoveryPipeline
        return DiscoveryPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")