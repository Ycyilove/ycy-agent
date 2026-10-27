# scripts/rebuild_mcp_intent_index.py
"""手动重建 MCP 意图索引（写 data/mcp_intent_index/）。"""
import asyncio
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from infra.offline import configure_hf_env
configure_hf_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _main():
    from backend.app import get_llm
    from mcp_client import MCPClientManager, load_mcp_config
    from mcp_client.dynamic_store import load_all_servers
    from backend.routes.mcp import _resolve_command
    from tools.mcp_intent_index import build_index

    configs = load_mcp_config()
    if not configs:
        logger.info("mcp_servers.json 为空，跳过")
        return

    manager = MCPClientManager(configs)

    def _noop_register(**kwargs):
        return None

    def _noop_unregister(name):
        return None

    try:
        for name, config in manager.configs.items():
            if not getattr(config, "enabled", True):
                logger.info("skip disabled server: %s", name)
                continue
            if config.transport == "stdio":
                if _resolve_command(config.command) is None:
                    logger.warning("skip %s: command not found (%s)", name, config.command)
                    continue
            try:
                await asyncio.wait_for(manager.connect(name), timeout=20.0)
                await asyncio.wait_for(
                    manager.discover(name, _noop_register, _noop_unregister),
                    timeout=20.0,
                )
                logger.info("connected: %s", name)
            except Exception as exc:
                logger.warning("connect %s failed, skip: %s: %s", name, type(exc).__name__, exc)

        result = build_index(load_all_servers(), manager, get_llm())
        logger.info("done: %r", result)
    finally:
        try:
            await manager.close()
        except Exception:
            logger.exception("manager.close failed")


if __name__ == "__main__":
    asyncio.run(_main())