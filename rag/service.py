"""Persistent local vector-store RAG service."""

import json
import logging
import os
import re
import uuid
from typing import Dict, List, Optional

# 全部改成相对 import：service.py 位于 rag/ 内部，
# 用 from rag import X 会和 rag/__init__.py 形成循环。
from .document_parser import DocumentParserFactory
from .vector_store import FAISSVectorStore
from .resources import ResourceManager
from .text_chunker import TextChunker
from .bm25_store import BM25Store, rrf_fuse
from .text_utils import clean_cn_spaces, clean_heading
from .entity.extractor import EntityExtractor
from .index.record_store import RecordStore
from .search.record_search import record_search
from .structure.parser import parse_structure

# LLM 走 core.llm.provider（Lazy 加载，首次调用才构造 ChatOpenAI）
from core.llm.provider import ModelScopeLLM

logger = logging.getLogger(__name__)

class LocalRAGService:
    """Ingest files, link extracted resources, and answer retrieval queries."""

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
        from .structure.field_classifier import FieldSchema
        schema_path = os.getenv('RAG_FIELD_SCHEMA', './config/field_schema.json')
        self._schema = FieldSchema.load(schema_path)
        self._extractor = EntityExtractor(
            dictionary=self._schema_dictionary(),
            patterns=self._schema_patterns(),
        )
        self.record_store = RecordStore(
            path=os.getenv('RAG_RECORD_STORE', './vector_store/records.json'),
            schema=self._schema,
        )

    def _schema_dictionary(self) -> Dict[str, str]:
        """从 schema 抽词典：{词: 类型}。"""
        out: Dict[str, str] = {}
        for name, td in self._schema.types.items():
            for term in td.dictionary:
                out[term] = name
        return out

    def _schema_patterns(self) -> Dict[str, str]:
        """从 schema 抽值模式：{类型: 正则}。多模式时取第一个。"""
        out: Dict[str, str] = {}
        for name, td in self._schema.types.items():
            if td.value_patterns:
                out[name] = td.value_patterns[0]
        return out

    def parse_tables_grid(
        tables: List[dict],
        source_filename: str,
        schema: Optional[FieldSchema] = None,
    ) -> List[Record]:
        """从 PyMuPDF extract_tables() 的 grid 建 Record。

        grid 是 [[cell_text, ...], ...]，第一行是表头。
        """
        if schema is None:
            schema = get_default_schema()

        class _Cell:
            def __init__(self, text):
                self.text = text

        class _Grid:
            def __init__(self, g):
                self.grid = g

        class _Table:
            def __init__(self, g, pages):
                self.data = _Grid(g)
                self.prov = [_Prov(p) for p in pages]

        class _Prov:
            def __init__(self, page_no):
                self.page_no = page_no

        records: List[Record] = []
        for t in tables:
            grid = [[_Cell(c) for c in row] for row in t["grid"]]
            records.extend(
                _table_to_records(
                    _Table(grid, t.get("page_numbers", [])),
                    source_filename, [], t.get("page_numbers", []), schema,
                )
            )
        return records

    def add_file(self, file_content: bytes, filename: str) -> Dict:
        """解析文件 → Docling 层级解析 → HybridChunker 切块 → 索引。"""
        import tempfile

        document_id = uuid.uuid4().hex
        resources_ingested = False
        tmp_path = None

        try:
            if self.vector_store.file_exists(filename):
                return {'status': 'warning', 'message': f'文件 {filename} 已存在，跳过添加'}

            # ── 1. 写入临时文件（Docling 需要磁盘路径）──
            ext = os.path.splitext(filename)[1] or ".pdf"
            fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="rag_")
            with os.fdopen(fd, "wb") as f:
                f.write(file_content)

            # ── 2. Docling 解析 + 切块 ──
            from .docling_parser import DoclingParser
            parser = DoclingParser()
            docling_chunks = parser.parse_to_chunks(tmp_path)
            if not docling_chunks:
                return {'status': 'error', 'message': 'Docling 解析后无 chunk'}

            # ── 2.1 结构解析：表格/列表 → Record ──
            if parser.last_document is not None:
                records = parse_structure(
                    parser.last_document, docling_chunks, filename,
                    schema=self._schema,
                )
            elif getattr(parser, "last_fast_tables", None):
                # docling 不可用时，用 PyMuPDF 识别出的表格建 Record
                from rag.structure.parser import parse_tables_grid
                records = parse_tables_grid(
                    parser.last_fast_tables, filename, schema=self._schema,
                )
            else:
                records = []

            # ── 2.5 Contextual Retrieval：为每个 chunk 生成情境化摘要 ──
            contextual_texts = None
            try:
                from backend.config import (
                    RAG_CONTEXTUAL_ENABLED,
                    RAG_CONTEXTUAL_CONCURRENCY,
                    RAG_CONTEXTUAL_TIMEOUT,
                    RAG_CONTEXTUAL_MIN_CHARS,
                )
                if RAG_CONTEXTUAL_ENABLED:
                    from .contextualizer import ChunkContextualizer

                    ctx_inputs = []
                    for dc in docling_chunks:
                        h_list = list(getattr(dc.meta, "headings", None) or [])
                        h_list = [clean_heading(h) for h in h_list]
                        ctx_inputs.append({
                            "text": clean_cn_spaces(dc.text or ""),
                            "headings": h_list,
                        })

                    ctx = ChunkContextualizer(
                        max_workers=RAG_CONTEXTUAL_CONCURRENCY,
                        timeout=RAG_CONTEXTUAL_TIMEOUT,
                        min_chunk_chars=RAG_CONTEXTUAL_MIN_CHARS,
                    )
                    ctx_results = ctx.contextualize_batch(ctx_inputs)
                    contextual_texts = [
                        r.get("enriched_text", "") for r in ctx_results
                    ]
                    n_ok = sum(1 for r in ctx_results if r.get("summary"))
                    logger.info(
                        "[dsh][rag] Contextual Retrieval: %d/%d chunk 生成摘要",
                        n_ok, len(ctx_results),
                    )
            except Exception:
                logger.exception(
                    "[dsh][rag] Contextual Retrieval 失败，降级为原文本"
                )
                contextual_texts = None

            # ── 3. 提取索引文本 + 元数据 ──
            indexed_texts = []
            metadata = []
            for i, dc in enumerate(docling_chunks):
                original = clean_cn_spaces(dc.text)

                # Contextual Retrieval：优先用摘要+原文，否则用 heading+原文
                if (
                    contextual_texts is not None
                    and i < len(contextual_texts)
                    and contextual_texts[i]
                ):
                    enriched = contextual_texts[i]
                else:
                    enriched = parser.get_enriched_text(dc)

                if not enriched or not enriched.strip():
                    continue
                enriched = clean_cn_spaces(enriched)

                headings = list(getattr(dc.meta, "headings", None) or [])
                headings = [clean_heading(h) for h in headings]
                heading_levels = list(
                    getattr(dc.meta, "heading_levels", None) or []
                )
                page_numbers = list(getattr(dc.meta, "page_numbers", None) or [])

                indexed_texts.append(enriched)
                metadata.append({
                    'chunk_id': uuid.uuid4().hex,
                    'document_id': document_id,
                    'source_filename': filename,
                    'headings': headings,
                    'page_numbers': page_numbers,
                    'original_text': original,
                    'has_contextual': bool(
                        contextual_texts is not None
                        and i < len(contextual_texts)
                        and contextual_texts[i]
                    ),
                    'heading_levels': heading_levels,
                })

            if not indexed_texts:
                return {'status': 'error', 'message': 'Docling 切块后内容为空'}

            # ── 4. 提取资源（图片/附件）──
            resources = self.resource_manager.ingest(
                content=file_content,
                filename=filename,
                document_id=document_id,
            )
            resources_ingested = True
            resource_ids = [r.id for r in resources]

            for m in metadata:
                m['resource_ids'] = resource_ids

            # ── 5. 写入向量库（用 enriched 文本）──
            result = self.vector_store.add_documents(
                indexed_texts,
                [filename] * len(indexed_texts),
                metadata=metadata,
            )
            if result.get('status') != 'success':
                self.resource_manager.delete_source(filename)
                return result

            # ── 5.5 记录级索引（表格/列表）──
            if records:
                self.record_store.add(records)

            # ── 6. 重建 BM25，关联资源 ──
            # 先 reset（清掉旧索引），再 build（构建新索引）
            self.bm25_store.reset()
            self.bm25_store.build(self.vector_store.metadata)
            self.resource_manager.repository.link_chunks(
                [m['chunk_id'] for m in metadata],
                resource_ids,
            )
            result['resources'] = [
                self.resource_manager.to_ref(r) for r in resources
            ]
            return result

        except ValueError as error:
            if resources_ingested:
                self.resource_manager.delete_source(filename)
            return {'status': 'error', 'message': str(error)}
        except Exception as error:
            if resources_ingested:
                self.resource_manager.delete_source(filename)
            logger.exception("[dsh][rag] 处理文件时出错")
            return {'status': 'error', 'message': f'处理文件时出错: {error}'}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def add_files(self, files: List[tuple]) -> Dict:
        results = []
        success_count = 0
        for file_content, filename in files:
            result = self.add_file(file_content, filename)
            results.append({'filename': filename, **result})
            if result['status'] == 'success':
                success_count += 1

        return {
            'status': 'success',
            'message': f'成功添加 {success_count} 个文件，失败 {len(files) - success_count} 个',
            'details': results,
        }

    def search(
        self,
        query: str,
        top_k: int = 15,
        rewrite: bool = True,
        use_bm25: bool = True,
        mode: str = "auto",
    ) -> List[Dict]:
        """向量检索 + BM25 混合检索 + 可选 query 改写。

        mode:
            - "flat"   ：原 chunk 级混合检索
            - "record" ：记录级检索（表格/列表优先）
            - "auto"   ：有记录且命中记录 BM25 时走 record，否则 flat
        """
        query = clean_cn_spaces(query)

        if mode == "record" or (mode == "auto" and self._should_use_record(query)):
            return self._record_search(query, top_k)

        # 1. query 改写（带超时降级）
        effective_query = query
        if rewrite:
            try:
                from backend.config import (
                    RAG_QUERY_REWRITE_ENABLED,
                    RAG_QUERY_REWRITE_TIMEOUT,
                )
                if RAG_QUERY_REWRITE_ENABLED:
                    from .query_rewriter import get_query_rewriter
                    import concurrent.futures

                    # 不能用 with：__exit__ 会 shutdown(wait=True)，
                    # 超时后仍会阻塞到改写跑完，超时形同虚设。
                    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        future = ex.submit(
                            get_query_rewriter().rewrite, query
                        )
                        effective_query = future.result(
                            timeout=RAG_QUERY_REWRITE_TIMEOUT
                        )
                    except concurrent.futures.TimeoutError:
                        logger.warning(
                            "[dsh][rag] query 改写超时(%ss)，用原 query",
                            RAG_QUERY_REWRITE_TIMEOUT,
                        )
                        effective_query = query
                    finally:
                        ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                logger.exception("[dsh][rag] query 改写失败，用原 query")

        pool_size = max(top_k * 3, top_k + 10)

        # 2. 向量检索：原 query 单路
        vector_results = self.vector_store.search(query, pool_size)

        # 3. BM25 检索：用实体 query（避免高频意图词稀释）
        bm25_results: List[Dict] = []
        if use_bm25:
            try:
                from backend.config import RAG_BM25_ENABLED, RAG_RRF_K
                if RAG_BM25_ENABLED:
                    if not self.bm25_store.is_built():
                        logger.info("[dsh][rag] BM25 索引未构建，惰性构建")
                        self.bm25_store.build(self.vector_store.metadata)

                    def _provider():
                        return self.vector_store.metadata

                    # 双 query：实体串 + 全句，两路都搜再融合。
                    # 不再用实体串替换全句，避免意图词（"获奖"/"团队"）被丢弃。
                    queries = self._bm25_queries_for(query)
                    if effective_query != query:
                        queries.extend(self._bm25_queries_for(effective_query))

                    seen = set()
                    uniq = []
                    for q in queries:
                        if q and q not in seen:
                            seen.add(q)
                            uniq.append(q)

                    lists = [
                        self.bm25_store.search(q, pool_size, _provider)
                        for q in uniq
                    ]
                    lists = [r for r in lists if r]
                    if len(lists) == 1:
                        bm25_results = lists[0]
                    elif len(lists) > 1:
                        bm25_results = rrf_fuse(
                            *lists, k=RAG_RRF_K, limit=pool_size,
                        )
            except Exception:
                logger.exception("[dsh][rag] BM25 检索失败，仅用向量结果")

        # 4. 融合
        if bm25_results:
            from backend.config import RAG_RRF_K
            fused = rrf_fuse(
                vector_results,
                bm25_results,
                k=RAG_RRF_K,
                limit=top_k,
                weights=[1.0, 1.5],
            )
        else:
            fused = vector_results[:top_k]

        # ── 返回前：从 metadata 取 headings ──
        # 关键：返回 index 里实际存的文本（enriched / contextual），
        # 而不是 original_text。否则会出现：
        #   向量/BM25 命中 enriched，但返回给用户的是 original，
        #   用户搜的词在返回文本里找不到。
        enriched = []
        for item in fused:
            meta = item.get("metadata", {})
            headings = meta.get("headings", [])
            indexed_text = item.get("text", "")

            # indexed_text 通常已带 heading 前缀（add_file 里 parser.get_enriched_text 产出），
            # 只有缺失时才补一次，避免重复前缀。
            if headings and not indexed_text.startswith(headings[0]):
                prefix = " > ".join(headings) + "\n"
                item["text"] = prefix + indexed_text
            else:
                item["text"] = indexed_text

            item["headings"] = headings
            item["heading_levels"] = meta.get("heading_levels", [])
            item["page_numbers"] = meta.get("page_numbers", [])
            item["has_contextual"] = meta.get("has_contextual", False)
            enriched.append(item)

        return self.resource_manager.enrich_search_results(enriched)

    def _bm25_queries_for(self, text: str) -> List[str]:
        """返回 [实体串, 全句]，两路分别检索后 RRF。

        与旧 _bm25_query_for 的区别：不再"二选一"，
        实体和意图词都保留在检索里。
        """
        ents = [m.surface for m in self._extractor.extract(text)]
        queries: List[str] = []
        if ents:
            queries.append(" ".join(ents))
        queries.append(text)
        return queries

    _RELATION_HINTS = ("属于", "队友", "成员", "团队", "关联", "同队", "所在", "获得")

    def _should_use_record(self, query: str) -> bool:
        """auto 模式判定：有记录 + 命中记录 BM25 或含关系词。"""
        if not self.record_store.stats().get("total_records"):
            return False
        if any(h in query for h in self._RELATION_HINTS):
            return True
        rec_hits = self.record_store.search(query, top_k=1)
        if not rec_hits:
            return False
        # 记录 BM25 命中即走 record；否则 flat 更稳
        return True

    def _record_search(self, query: str, top_k: int) -> List[Dict]:
        """记录级检索 → 转成与 flat search 同构的 dict 列表。"""
        result = record_search(
            query, self.record_store, self._extractor,
            top_k=top_k, schema=self._schema,
        )
        docs: List[Dict] = []
        for record in result["records"]:
            docs.append({
                "text": record.to_index_text(),
                "filename": record.source_filename,
                "score": 0.0,
                "metadata": {
                    "chunk_id": record.record_id,
                    "record_id": record.record_id,
                    "source_filename": record.source_filename,
                    "headings": record.headings,
                    "page_numbers": record.page_numbers,
                    "kind": "record",
                },
            })
        enriched = self.resource_manager.enrich_search_results(docs)
        # 把路径和警告挂在每条结果上，供 _build_context / prompt 使用
        for item in enriched:
            item["paths"] = result["paths"]
            item["warnings"] = result["warnings"]
        return enriched

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        context_parts = []
        for index, doc in enumerate(retrieved_docs, 1):
            source = doc.get('filename', '未知来源')
            content = doc.get('text', '')

            resource_lines = []
            for resource in doc.get('resources', []):
                location = (
                    f"，第 {resource['page_no']} 页"
                    if resource.get('page_no') else ''
                )
                resource_lines.append(
                    f"[[asset:{resource['id']}]] "
                    f"{resource['kind']}: {resource['filename']}{location}"
                )

            if resource_lines:
                joined = '\n'.join(resource_lines)
                resource_text = f'\n关联资源:\n{joined}'
            else:
                resource_text = ''

            context_parts.append(
                f'【文档 {index}】来源: {source}\n{content}{resource_text}'
            )
        return '\n\n'.join(context_parts)

    @staticmethod
    def _build_prompt(query: str, context: str, allow_asset_tags: bool = True) -> str:
        citation_rules = (
            '如果需要引用图片或附件，只能使用参考信息中的 [[asset:资源ID]] 标记，不要编造 URL。'
            if allow_asset_tags
            else '资源会由系统单独返回，请不要输出资源标记或文件路径。'
        )
        return f'''基于以下参考信息回答用户问题。

参考信息:
{context}

回答规则:
1. 只使用参考信息回答，不要使用你的通用知识。
2. 如果参考信息不足以回答，请明确说"知识库中没有找到相关信息"，不要编造。
3. {citation_rules}
4. 不要输出服务器本地路径。

用户问题: {query}

回答:'''

    def rag_answer(self, query: str, top_k: int = 3) -> Dict:
        retrieved_docs = self.search(query, top_k)
        if not retrieved_docs:
            return {
                'answer': '知识库为空，请先上传文档',
                'retrieved_docs': [],
                'resources': [],
                'context': '',
            }

        context = self._build_context(retrieved_docs)
        resources = self.resource_manager.collect_refs(retrieved_docs)
        answer = ModelScopeLLM().generate([{'role': 'user', 'content': self._build_prompt(query, context)}])
        answer = self.resource_manager.resolve_asset_tags(answer, resources)
        return {
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'resources': resources,
            'context': context,
        }

    def stream_rag_answer(self, query: str, top_k: int = 3):
        retrieved_docs = self.search(query, top_k)
        if not retrieved_docs:
            yield f"data: {json.dumps({'text': '知识库为空，请先上传文档'})}\n\n"
            return

        context = self._build_context(retrieved_docs)
        resources = self.resource_manager.collect_refs(retrieved_docs)
        yield f"data: {json.dumps({'type': 'resources', 'resources': resources}, ensure_ascii=False)}\n\n"
        prompt = self._build_prompt(query, context, allow_asset_tags=False)
        yield from ModelScopeLLM().stream_generate([{'role': 'user', 'content': prompt}])

    def delete_file(self, filename: str) -> Dict:
        result = self.vector_store.delete_file(filename)
        if result.get('status') == 'success':
            result['deleted_resources'] = self.resource_manager.delete_source(filename)
            result['deleted_records'] = self.record_store.delete_by_source(filename)
            self.bm25_store.reset()
        return result

    def clear_all(self) -> Dict:
        result = self.vector_store.clear_all()
        self.resource_manager.clear()
        self.record_store.clear()
        self.bm25_store.reset()
        result.update({'resources': 'cleared', 'records': 'cleared'})
        return result

    def get_stats(self) -> Dict:
        result = self.vector_store.get_stats()
        result.update(self.resource_manager.stats())
        result.update(self.record_store.stats())
        return result
