"""SentenceTransformer 统一入口（进程内单例 + 本地路径探测）。

解决的问题：
    - vector_store 用 bge-small-zh-v1.5，tool_shortlister 用 all-MiniLM-L6-v2，
      mem0 内部又自己构造一个 embedder，三处各自联网、各自占内存。
    - 硬编码路径换机器就失效（此前 D:\\Projects\\111\\... 的错误）。
    - 本地探测逻辑重复写在 vector_store._init_embeddings 里。

设计：
    - get_encoder(model_name) 用 lru_cache 缓存实例，同模型只加载一次。
    - resolve_local_model_dir 统一探测 HF cache 格式 + 扁平目录。
    - 默认 local_files_only=True，本地找不到才回退 hub；调用方可显式关掉。
    - 与 infra.offline 配合：本地命中时在 offline_mode() 里加载，杜绝联网校验。

不适用范围：
    - mem0 的 embedder 由 Memory.from_config 内部构造，无法直接注入本单例；
      mem0 侧靠 offline_mode() + 预热（先调用一次 get_encoder）命中同一份磁盘 cache。
"""

from __future__ import annotations

import glob
import logging
import os
from functools import lru_cache
from typing import Optional

from infra.offline import offline_mode

__all__ = [
    "resolve_local_model_dir",
    "get_encoder",
    "clear_encoder_cache",
]

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


def resolve_local_model_dir(model_name: str) -> Optional[str]:
    """探测本地模型目录，返回含 config.json 的目录，找不到返回 None。

    优先级：
        1) HF cache：models/models--<org>--<name>/snapshots/<hash>/
        2) 扁平目录：models/<name 最后一段>/
        3) 全名目录：models/<org>/<name>/ 或 models/<org>--<name>/
    """
    if not model_name:
        return None

    # 1) HF cache 格式
    hf_dir = os.path.join(_MODELS_DIR, f"models--{model_name.replace('/', '--')}")
    if os.path.isdir(hf_dir):
        snaps = sorted(
            glob.glob(os.path.join(hf_dir, "snapshots", "*")),
            key=os.path.getmtime,
            reverse=True,
        )
        for snap in snaps:
            if os.path.isfile(os.path.join(snap, "config.json")):
                return snap

    # 2) 扁平目录（只取最后一段，如 bge-small-zh-v1.5）
    flat = os.path.join(_MODELS_DIR, model_name.split("/")[-1])
    if os.path.isfile(os.path.join(flat, "config.json")):
        return flat

    # 3) 全名 / org--name 目录
    for cand in (
        os.path.join(_MODELS_DIR, model_name),
        os.path.join(_MODELS_DIR, model_name.replace("/", "--")),
    ):
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand

    return None


@lru_cache(maxsize=4)
def _load_encoder(model_name: str, local_files_only: bool):
    """实际加载函数；被 get_encoder 缓存。"""
    from sentence_transformers import SentenceTransformer

    local_dir = resolve_local_model_dir(model_name) if local_files_only else None
    if local_dir:
        try:
            with offline_mode():
                enc = SentenceTransformer(local_dir, local_files_only=True)
            dim = enc.get_sentence_embedding_dimension()
            logger.info("[embedding] local hit: %s (dim=%d)", local_dir, dim)
            return enc
        except Exception as exc:
            logger.warning(
                "[embedding] local dir exists but load failed (%s: %s), fallback to hub",
                type(exc).__name__, exc,
            )

    if local_files_only:
        logger.warning(
            "[embedding] local miss for %s, fallback to hub (may hit network)",
            model_name,
        )
    enc = SentenceTransformer(model_name)
    dim = enc.get_sentence_embedding_dimension()
    logger.info("[embedding] hub load: %s (dim=%d)", model_name, dim)
    return enc


def get_encoder(model_name: str, local_files_only: bool = True):
    """获取（缓存的）SentenceTransformer 实例。

    Args:
        model_name: 如 "BAAI/bge-small-zh-v1.5"。
        local_files_only: True 时优先本地，本地缺失才回退 hub。

    Returns:
        SentenceTransformer 实例（同参数返回同一对象）。
    """
    return _load_encoder(model_name, local_files_only)


def clear_encoder_cache() -> None:
    """清空缓存（测试用；生产慎用，会释放模型内存）。"""
    _load_encoder.cache_clear()