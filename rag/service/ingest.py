"""入库服务：add_file / add_files。

B14-2 从 rag/service.py 原样搬（add_file 281 行 + add_files 15 行）。
IngestService.__init__ 做引用转发（self.vector_store = rag.vector_store 等），
所以 add_file 方法体 self.xxx 前缀零改动。
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Dict, List, Optional
from pathlib import Path

from rag.document_parser import DocumentParserFactory
from rag.text_utils import clean_cn_spaces, clean_heading
from rag.structure.parser import parse_structure

logger = logging.getLogger(__name__)


class IngestService:
    """入库服务。self.xxx 是 rag.xxx 的引用转发（零改动搬 add_file）。"""

    def __init__(self, rag):
        self.rag = rag
        # 引用转发：self.xxx 和 rag.xxx 指向同一对象
        self.vector_store = rag.vector_store
        self.bm25_store = rag.bm25_store
        self.resource_manager = rag.resource_manager
        self.record_store = rag.record_store
        self.chunker = rag.chunker
        self._schema = rag._schema
        self._extractor = rag._extractor
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

            # ── 2.1 统一层级纠正（无论底层走 Docling 还是 FastPath） ──
            try:
                from .structure.corrector import HierarchyCorrector
                corrector = HierarchyCorrector()
                docling_chunks = corrector.correct_chunks(docling_chunks)
            except Exception:
                logger.exception("[dsh][rag] 层级纠正失败，继续使用原始层级")

            # ── 2.2 结构解析：表格/列表 → Record ──
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

                headings = list(getattr(dc.meta, "headings", None) or [])
                headings = [clean_heading(h) for h in headings]
                heading_levels = list(getattr(dc.meta, "heading_levels", None) or [])
                page_numbers = list(getattr(dc.meta, "page_numbers", None) or [])

                # 强制构建 Breadcrumb，防止 Contextual 分支覆盖
                breadcrumb = ""
                if headings:
                    breadcrumb = " > ".join(headings) + "\n"

                if (
                    contextual_texts is not None
                    and i < len(contextual_texts)
                    and contextual_texts[i]
                ):
                    # 摘要 + 纠正后的上下文前缀
                    enriched = breadcrumb + contextual_texts[i]
                else:
                    enriched = parser.get_enriched_text(dc)
                    if breadcrumb and not enriched.startswith(headings[0]):
                        enriched = breadcrumb + enriched

                if not enriched or not enriched.strip():
                    continue
                enriched = clean_cn_spaces(enriched)

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

            # pypdf 没提出图，但 Docling 有 pictures → 补提
            if not resources and parser.last_document is not None:
                try:
                    from .docling_parser import extract_pictures_from_doc
                    from .resources import ResourceRecord
                    import hashlib, uuid as _uuid

                    docling_resources = extract_pictures_from_doc(
                        parser.last_document, filename, document_id,
                    )
                    for parsed in docling_resources:
                        rid = _uuid.uuid4().hex
                        storage_key = self.resource_manager.store.put(
                            resource_id=rid,
                            filename=parsed.filename,
                            content=parsed.content,
                        )
                        record = ResourceRecord(
                            id=rid,
                            document_id=document_id,
                            source_filename=filename,
                            filename=parsed.filename,
                            mime_type=parsed.mime_type,
                            storage_key=storage_key,
                            sha256=hashlib.sha256(parsed.content).hexdigest(),
                            kind=parsed.kind.value,
                            page_no=parsed.page_no,
                        )
                        self.resource_manager.repository.save(record)
                        resources.append(record)
                    if docling_resources:
                        logger.info(
                            "[dsh][rag] pypdf 无资源，Docling 补提 %d 个",
                            len(docling_resources),
                        )
                except Exception:
                    logger.exception("[dsh][rag] Docling 补提图片失败")

            resources_ingested = True
            resource_ids = [r.id for r in resources]

            # ── 按页分配 resource_ids：每个 chunk 只关联自己页码覆盖的资源 ──
            # 建立 page_no -> [resource_id] 映射
            page_to_rids: Dict[int, List[str]] = {}
            no_page_rids: List[str] = []
            for r in resources:
                if r.page_no is None:
                    no_page_rids.append(r.id)
                else:
                    page_to_rids.setdefault(r.page_no, []).append(r.id)

            for m in metadata:
                chunk_pages = m.get("page_numbers") or []
                if chunk_pages:
                    # chunk 有页码 → 只取覆盖页的资源（±1 容差）
                    allowed = set()
                    for p in chunk_pages:
                        if isinstance(p, int):
                            allowed.update({p - 1, p, p + 1})
                    chunk_rids: List[str] = []
                    for pg in allowed:
                        chunk_rids.extend(page_to_rids.get(pg, []))
                    # 无页码的资源（附件类）也带上
                    chunk_rids.extend(no_page_rids)
                    # 去重保序
                    seen = set()
                    m['resource_ids'] = [
                        rid for rid in chunk_rids
                        if not (rid in seen or seen.add(rid))
                    ]
                else:
                    # chunk 无页码 → 不写 resource_ids，让 source_filename 兜底
                    m['resource_ids'] = []

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
            # 注意：不在这里 build，让 search() 惰性构建（P2-1 已应用）
            
            # 主动预热：入库后立即构建一次并写缓存，
            # 下次 search() 命中缓存，跳过 jieba 分词。
            try:
                self.bm25_store.build(self.vector_store.metadata)
            except Exception:
                logger.exception("[dsh][rag] BM25 预热失败（非致命）")

            # 批量版：一次性收集所有 (chunk_id, rids)，调一次 link_chunks
            links_map: Dict[str, List[str]] = {}
            for m in metadata:
                rids = m.get('resource_ids') or []
                if rids:
                    links_map[m['chunk_id']] = rids

            if links_map:
                # 直接操作 repository 内部，避免多次 _persist
                with self.resource_manager.repository._lock:
                    self.resource_manager.repository._links.update(links_map)
                    self.resource_manager.repository._persist()
                    
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

