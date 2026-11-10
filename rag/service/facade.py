"""LocalRAGService 组合类（B14-2 重写）。

公共 API 与 rag/service.py 完全一致，1 行委托到 4 个协作类：
    - IngestService:    add_file / add_files
    - SearchService:    search / _bm25_queries_for / _should_use_record / _record_search
    - AnswerService:    rag_answer / stream_rag_answer / _build_context / _build_prompt
    - LifecycleService: delete_file / clear_all / get_stats

B14-2 只建 ingest.py，B14-3 建 search.py / answer.py / lifecycle.py。
现在 3 个协作类还不在，facade 的对应方法会 ImportError（延迟 import），
但旧 rag/service.py 仍优先（Python 文件 > 目录），所以不影响。
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Dict, List, Optional
from pathlib import Path

from rag.document_parser import DocumentParserFactory
from rag.vector_store import FAISSVectorStore
from rag.resources import ResourceManager
from rag.text_chunker import TextChunker
from rag.bm25_store import BM25Store, rrf_fuse
from rag.text_utils import clean_cn_spaces, clean_heading
from rag.entity.extractor import EntityExtractor
from rag.index.record_store import RecordStore
from rag.search.record_search import record_search
from rag.structure.parser import parse_structure

logger = logging.getLogger(__name__)


class LocalRAGService:
    """本地 RAG 服务。B14 拆包后保留为组合类，公共 API 不变。"""

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        resource_dir: Optional[str] = None,
        resource_metadata_path: Optional[str] = None,
    ):
        if embedding_model is None:
            try:
                from backend.config import RAG_EMBEDDING_MODEL
                embedding_model = RAG_EMBEDDING_MODEL
            except ImportError:
                embedding_model = 'BAAI/bge-small-zh-v1.5'
        self.embedding_model = embedding_model
        self.vector_store = FAISSVectorStore(embedding_model=embedding_model)
        self.bm25_store = BM25Store()
        self.chunker = TextChunker(chunk_size=500, overlap=100)
        self.resource_manager = ResourceManager(
            root=resource_dir or os.getenv('RAG_RESOURCE_DIR', './rag_resources'),
            metadata_path=(
                resource_metadata_path
                or os.getenv('RAG_RESOURCE_METADATA', './vector_store/resources.json')
            ),
        )
        from rag.structure.field_classifier import FieldSchema
        _SCHEMA_PATH = Path(__file__).resolve().parent.parent / "config" / "field_schema.json"
        schema_path = os.getenv("RAG_FIELD_SCHEMA", str(_SCHEMA_PATH))
        self._schema = FieldSchema.load(schema_path)

        try:
            from rag.structure.field_classifier import set_default_schema
            set_default_schema(self._schema)
        except Exception:
            logger.exception("[dsh][rag] set_default_schema 失败")
        self._extractor = EntityExtractor(
            dictionary=self._schema_dictionary(),
            patterns=self._schema_patterns(),
        )
        self.record_store = RecordStore(
            path=os.getenv('RAG_RECORD_STORE', './vector_store/records.json'),
            schema=self._schema,
        )

        # ── 4 个协作类（B14-2 只建 ingest，B14-3 建其余 3 个）──
        from .ingest import IngestService
        self.ingest = IngestService(self)

    def _schema_dictionary(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name, td in self._schema.types.items():
            for term in td.dictionary:
                out[term] = name
        return out

    def _schema_patterns(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name, td in self._schema.types.items():
            if td.value_patterns:
                out[name] = td.value_patterns[0]
        return out

    # ── 公共方法：1 行委托 ──

    def add_file(self, file_content: bytes, filename: str) -> Dict:
        return self.ingest.add_file(file_content, filename)

    def add_files(self, files: List[tuple]) -> Dict:
        return self.ingest.add_files(files)

    def search(
        self,
        query: str,
        top_k: int = 8,
        rewrite: bool = True,
        use_bm25: bool = True,
        mode: str = "auto",
    ) -> List[Dict]:
        from .search import SearchService
        if not hasattr(self, "_search_svc"):
            self._search_svc = SearchService(self)
        return self._search_svc.search(query, top_k, rewrite=rewrite, use_bm25=use_bm25, mode=mode)

    def _bm25_queries_for(self, text: str) -> List[str]:
        from .search import SearchService
        if not hasattr(self, "_search_svc"):
            self._search_svc = SearchService(self)
        return self._search_svc.bm25_queries_for(text)

    def _should_use_record(self, query: str) -> bool:
        from .search import SearchService
        if not hasattr(self, "_search_svc"):
            self._search_svc = SearchService(self)
        return self._search_svc.should_use_record(query)

    def _record_search(self, query: str, top_k: int) -> List[Dict]:
        from .search import SearchService
        if not hasattr(self, "_search_svc"):
            self._search_svc = SearchService(self)
        return self._search_svc.record_search_docs(query, top_k)

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        from .answer import AnswerService
        if not hasattr(self, "_answer_svc"):
            self._answer_svc = AnswerService(self)
        return self._answer_svc.build_context(retrieved_docs)

    @staticmethod
    def _build_prompt(query: str, context: str, allow_asset_tags: bool = True) -> str:
        from .answer import AnswerService
        return AnswerService.build_prompt(query, context, allow_asset_tags=allow_asset_tags)

    def rag_answer(self, query: str, top_k: int = 3) -> Dict:
        from .answer import AnswerService
        if not hasattr(self, "_answer_svc"):
            self._answer_svc = AnswerService(self)
        return self._answer_svc.rag_answer(query, top_k)

    def stream_rag_answer(self, query: str, top_k: int = 3):
        from .answer import AnswerService
        if not hasattr(self, "_answer_svc"):
            self._answer_svc = AnswerService(self)
        return self._answer_svc.stream_rag_answer(query, top_k)

    def delete_file(self, filename: str) -> Dict:
        from .lifecycle import LifecycleService
        if not hasattr(self, "_lifecycle_svc"):
            self._lifecycle_svc = LifecycleService(self)
        return self._lifecycle_svc.delete_file(filename)

    def clear_all(self) -> Dict:
        from .lifecycle import LifecycleService
        if not hasattr(self, "_lifecycle_svc"):
            self._lifecycle_svc = LifecycleService(self)
        return self._lifecycle_svc.clear_all()

    def get_stats(self) -> Dict:
        from .lifecycle import LifecycleService
        if not hasattr(self, "_lifecycle_svc"):
            self._lifecycle_svc = LifecycleService(self)
        return self._lifecycle_svc.get_stats()