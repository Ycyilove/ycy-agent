"""HuggingFace 离线模式上下文管理。

统一管理 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_ENDPOINT / HF_HOME，
避免各模块（vector_store / mem0_layer / tool_shortlister）各写各的
try/finally，导致某一处漏恢复、或某处没设离线而联网。

用法：
    from infra.offline import offline_mode, configure_hf_env

    configure_hf_env()                      # 模块 import 时调一次，设 endpoint/home
    with offline_mode():                    # 需要强制离线的临界区
        model = SentenceTransformer(local_path, local_files_only=True)
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator, Optional

__all__ = ["configure_hf_env", "offline_mode", "hf_env_snapshot"]

_OFFLINE_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

# 项目根 = LLM/LLM（本文件在 infra/ 下）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_HF_HOME = os.path.join(_PROJECT_ROOT, "models")


def configure_hf_env(
    endpoint: Optional[str] = "https://hf-mirror.com",
    hf_home: Optional[str] = None,
) -> None:
    """设置 HF 环境变量（幂等，用 setdefault 不覆盖已有值）。

    应在任何 huggingface_hub / sentence_transformers import 之前调用。
    vector_store.py 顶部已有一份等价逻辑，后续可改为调用本函数。
    """
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)
    os.environ.setdefault("HF_HOME", hf_home or _DEFAULT_HF_HOME)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"])


def hf_env_snapshot() -> dict:
    """返回当前 HF 相关环境变量快照（调试用）。"""
    keys = ("HF_ENDPOINT", "HF_HOME", "HUGGINGFACE_HUB_CACHE", *_OFFLINE_KEYS)
    return {k: os.environ.get(k) for k in keys}


@contextlib.contextmanager
def offline_mode() -> Iterator[None]:
    """强制 HF 离线；退出时精确恢复到进入前的状态。

    进入前未设置的 key，退出时会被 pop 掉（而不是留空字符串），
    避免 "HF_HUB_OFFLINE=" 这种空值被下游误判为已设置。
    """
    backup = {k: os.environ.get(k) for k in _OFFLINE_KEYS}
    try:
        for k in _OFFLINE_KEYS:
            os.environ[k] = "1"
        yield
    finally:
        for k, v in backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v