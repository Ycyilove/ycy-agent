# tests/conftest.py
import os
import shutil
import tempfile
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYTEST_TMP_BASE = _PROJECT_ROOT / ".pytest_tmp"


@pytest.fixture
def tmp_path():
    """覆盖 pytest 默认 tmp_path，指到项目内 .pytest_tmp/，不占 C:。

    每个测试一个独立子目录，测试结束自动删。
    """
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
    否则首次用 rag_service 的测试会把 ~100s 初始化开销
    记到自己的 elapsed 里，导致超时类测试误判。
    """
    from rag import LocalRAGService
    svc = LocalRAGService()
    try:
        svc.search("warmup", top_k=1, rewrite=False)
    except Exception:
        # warmup 失败不影响测试；让后续测试自己暴露真实错误
        pass
    return svc
