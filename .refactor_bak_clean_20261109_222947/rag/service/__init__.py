"""LocalRAGService 拆分包（B14）。

对外只暴露 LocalRAGService（facade.py）。
子包：
    - ingest/    add_file / add_files 的 11 步
    - search/    search 的向量+BM25+RRF
    - answer/    rag_answer / stream_rag_answer
    - lifecycle  delete_file / clear_all / get_stats
"""
from .facade import LocalRAGService

__all__ = ["LocalRAGService"]