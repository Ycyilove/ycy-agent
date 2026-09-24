"""MCP 配置文件读写。程序独占管理，用户不手动改。"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def _config_path() -> Path:
    path = (
        os.getenv("MCP_CONFIG_FILE")
        or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "mcp_servers.json",
        )
    )
    return Path(path)


def load_all_servers() -> Dict[str, dict]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.exception("读取 MCP 配置失败: %s", path)
        return {}

    servers = data.get("servers") if isinstance(data, dict) else None
    return servers if isinstance(servers, dict) else {}


def _write_all(servers: Dict[str, dict]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"servers": servers}, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def save_server(name: str, mapping: dict) -> None:
    servers = load_all_servers()
    servers[name] = mapping
    _write_all(servers)


def delete_server(name: str) -> bool:
    servers = load_all_servers()
    if name not in servers:
        return False
    servers.pop(name)
    _write_all(servers)
    return True


def set_server_enabled(name: str, enabled: bool) -> bool:
    servers = load_all_servers()
    if name not in servers:
        return False
    entry = dict(servers[name])
    entry["enabled"] = bool(enabled)
    servers[name] = entry
    _write_all(servers)
    return True