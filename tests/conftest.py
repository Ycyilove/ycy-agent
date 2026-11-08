# tests/conftest.py
import os
import shutil
import tempfile
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYTEST_TMP_BASE = _PROJECT_ROOT / ".pytest_tmp"


def pytest_configure(config):
    """注册 asyncio marker，消除 PytestUnknownMarkWarning。"""
    config.addinivalue_line(
        "markers", "asyncio: async test (requires pytest-asyncio)"
    )


@pytest.fixture
def tmp_path():
    """覆盖 pytest 默认 tmp_path，指到项目内 .pytest_tmp/，不占 C:。"""
    _PYTEST_TMP_BASE.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=_PYTEST_TMP_BASE))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def rag_service():
    """session 级 RAG 实例。

    warmup 一次：加载 embedding 模型 + 构建 BM25 索引。
    """
    from rag import LocalRAGService
    svc = LocalRAGService()
    try:
        svc.search("warmup", top_k=1, rewrite=False)
    except Exception:
        pass
    return svc