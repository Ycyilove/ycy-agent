"""Persistent local vector-store RAG service."""

import json
import os
import uuid
from typing import Dict, List, Optional

from rag import DocumentParserFactory, FAISSVectorStore, ResourceManager, TextChunker

from .llm import ModelScopeLLM


class LocalRAGService:
    """Ingest files, link extracted resources, and answer retrieval queries."""

    def __init__(
        self,
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        resource_dir: Optional[str] = None,
        resource_metadata_path: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = FAISSVectorStore(embedding_model=embedding_model)
        self.chunker = TextChunker(chunk_size=500, overlap=100)
        self.resource_manager = ResourceManager(
            root=resource_dir or os.getenv('RAG_RESOURCE_DIR', './rag_resources'),
            metadata_path=(
                resource_metadata_path
                or os.getenv('RAG_RESOURCE_METADATA', './vector_store/resources.json')
            ),
        )

    def add_file(self, file_content: bytes, filename: str) -> Dict:
        document_id = uuid.uuid4().hex
        resources_ingested = False
        try:
            if self.vector_store.file_exists(filename):
                return {'status': 'warning', 'message': f'文件 {filename} 已存在，跳过添加'}

            text = DocumentParserFactory.parse_file(file_content, filename)
            if not text or not text.strip():
                return {'status': 'error', 'message': '文件内容为空'}

            chunks = self.chunker.chunk_text(text)
            if not chunks:
                return {'status': 'error', 'message': '文本分块失败'}

            resources = self.resource_manager.ingest(
                content=file_content,
                filename=filename,
                document_id=document_id,
            )
            resources_ingested = True
            resource_ids = [resource.id for resource in resources]
            chunk_ids = [uuid.uuid4().hex for _ in chunks]
            metadata = [
                {
                    'chunk_id': chunk_id,
                    'document_id': document_id,
                    'source_filename': filename,
                    'resource_ids': resource_ids,
                }
                for chunk_id in chunk_ids
            ]

            result = self.vector_store.add_documents(chunks, [filename] * len(chunks), metadata=metadata)
            if result.get('status') != 'success':
                self.resource_manager.delete_source(filename)
                return result

            self.resource_manager.repository.link_chunks(chunk_ids, resource_ids)
            result['resources'] = [self.resource_manager.to_ref(resource) for resource in resources]
            return result
        except ValueError as error:
            if resources_ingested:
                self.resource_manager.delete_source(filename)
            return {'status': 'error', 'message': str(error)}
        except Exception as error:
            if resources_ingested:
                self.resource_manager.delete_source(filename)
            return {'status': 'error', 'message': f'处理文件时出错: {error}'}

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

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        return self.resource_manager.enrich_search_results(self.vector_store.search(query, top_k))

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        context_parts = []
        for index, doc in enumerate(retrieved_docs, 1):
            source = doc.get('filename', '未知来源')
            content = doc.get('text', '')
            resource_lines = []
            for resource in doc.get('resources', []):
                location = f"，第 {resource['page_no']} 页" if resource.get('page_no') else ''
                resource_lines.append(
                    f"[[asset:{resource['id']}]] {resource['kind']}: {resource['filename']}{location}"
                )
            resource_text = f"\n关联资源:\n{'\n'.join(resource_lines)}" if resource_lines else ''
            context_parts.append(f'【文档 {index}】来源: {source}\n{content}{resource_text}')
        return '\n\n'.join(context_parts)

    @staticmethod
    def _build_prompt(query: str, context: str, allow_asset_tags: bool = True) -> str:
        citation_rules = (
            '如果需要引用图片或附件，只能使用参考信息中的 [[asset:资源ID]] 标记，不要编造 URL。'
            if allow_asset_tags
            else '资源会由系统单独返回，请不要输出资源标记或文件路径。'
        )
        return f'''基于以下参考信息回答用户问题。如果参考信息不足以回答，请基于你的知识回答，并说明情况。

参考信息:
{context}

回答规则:
1. 优先使用参考信息回答。
2. {citation_rules}
3. 不要输出服务器本地路径。

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
        return result

    def clear_all(self) -> Dict:
        result = self.vector_store.clear_all()
        self.resource_manager.clear()
        result.update({'resources': 'cleared'})
        return result

    def get_stats(self) -> Dict:
        result = self.vector_store.get_stats()
        result.update(self.resource_manager.stats())
        return result
