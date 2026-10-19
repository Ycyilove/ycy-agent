"""Persistent local vector-store RAG service."""

import json
import logging
import os
import re
import uuid
from typing import Dict, List, Optional

from rag import (
    DocumentParserFactory,
    FAISSVectorStore,
    ResourceManager,
    TextChunker,
    BM25Store,
    rrf_fuse,
)
from rag.text_utils import clean_cn_spaces, clean_heading

from .llm import ModelScopeLLM

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
            ext = os..path.splitext(filename)[1] or ".pdf"
            fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="rag_")
            with os..fdopen(fd, "wb") as f:
                f.write(file_content)

            # ── 2. Docling 解析 + 切块 ──
            from rag import DoclingParser
            parser = DoclingParser()
            docling_chunks = parser.parse_to_chunks(tmp_path)
            if not docling_chunks:
                return {'status': 'error', 'message': 'Docling 解析后无 chunk'}

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
                    from rag.contextualizer import ChunkContextualizer

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
            if tmp_path and os..path.exists(tmp_path):
                try:
                    os..unlink(tmp_path)
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
    ) -> List[Dict]:
        """向量检索 + BM25 混合检索 + 可选 query 改写。

        设计：
            - 向量检索用原 query（语义匹配）
            - BM25 用"实体 query"（精确匹配，避免高频意图词稀释实体）
            - 融合时 BM25 权重 1.5
        """
        query = clean_cn_spaces(query)

        # 1. query 改写（带超时降级）
        effective_query = query
        if rewrite:
            try:
                from backend.config import (
                    RAG_QUERY_REWRITE_ENABLED,
                    RAG_QUERY_REWRITE_TIMEOUT,
                )
                if RAG_QUERY_REWRITE_ENABLED:
                    from rag import get_query_rewriter
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
                from backend.config import RAG_BM25_ENABLED
                if RAG_BM25_ENABLED:
                    # 索引未构建时才惰性构建（避免每次查询都重建）
                    if not self.bm25_store.is_built():
                        logger.info("[dsh][rag] BM25 索引未构建，惰性构建")
                        self.bm25_store.build(self.vector_store.metadata)

                    from rag.query_rewriter import extract_entities

                    def _provider():
                        return self.vector_store.metadata

                    def _bm25_query_for(text: str) -> str:
                        ents = extract_entities(text)
                        return " ".join(ents) if ents else text

                    bm25_query = _bm25_query_for(query)
                    if bm25_query != query:
                        logger.info(
                            "[dsh][rag] BM25 query 用实体: %r → %r",
                            query, bm25_query,
                        )

                    eff_bm25_query = (
                        _bm25_query_for(effective_query)
                        if effective_query != query
                        else bm25_query
                    )

                    if eff_bm25_query == bm25_query:
                        # 实体相同：只搜一次，避免重复结果 + 双倍开销
                        bm25_results = self.bm25_store.search(
                            bm25_query, pool_size, _provider,
                        )
                    else:
                        # 实体不同：两次结果用 RRF 融合，不直接拼接
                        from backend.config import RAG_RRF_K
                        r1 = self.bm25_store.search(
                            bm25_query, pool_size, _provider,
                        )
                        r2 = self.bm25_store.search(
                            eff_bm25_query, pool_size, _provider,
                        )
                        bm25_results = rrf_fuse(
                            r1, r2, k=RAG_RRF_K, limit=pool_size,
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
            item["page_numbers"] = meta.get("page_numbers", [])
            item["has_contextual"] = meta.get("has_contextual", False)
            enriched.append(item)

        return self.resource_manager.enrich_search_results(enriched)

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
            self.bm25_store.reset()
        return result

    def clear_all(self) -> Dict:
        result = self.vector_store.clear_all()
        self.resource_manager.clear()
        self.bm25_store.reset()
        result.update({'resources': 'cleared'})
        return result

    def get_stats(self) -> Dict:
        result = self.vector_store.get_stats()
        result.update(self.resource_manager.stats())
        return result
