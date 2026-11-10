"""回答服务：build_context / build_prompt / rag_answer / stream_rag_answer。

B14-3 从 rag/service.py 原样搬（_build_context 31 + _build_prompt 21 + rag_answer 21 + stream_rag_answer 12）。
AnswerService.__init__ 做引用转发（self.xxx = rag.xxx）。
方法名映射（对齐 facade）：
    _build_context -> build_context
    _build_prompt -> build_prompt（@staticmethod）
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from core.llm.provider import ModelScopeLLM

logger = logging.getLogger(__name__)


class AnswerService:
    """回答服务。self.xxx 是 rag.xxx 的引用转发。"""

    def __init__(self, rag):
        self.rag = rag
        self.resource_manager = rag.resource_manager
        self.vector_store = rag.vector_store
        self.record_store = rag.record_store
    def build_context(self, retrieved_docs: List[Dict]) -> str:
        context_parts = []
        for index, doc in enumerate(retrieved_docs, 1):
            source = doc.get('filename', '未知来源')
            content = doc.get('text', '')
            if len(content) > 1500:
                content = content[:1500] + "...(truncated)"

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

    def build_prompt(query: str, context: str, allow_asset_tags: bool = True) -> str:
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

        context = self.build_context(retrieved_docs)
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

        context = self.build_context(retrieved_docs)
        resources = self.resource_manager.collect_refs(retrieved_docs)
        yield f"data: {json.dumps({'type': 'resources', 'resources': resources}, ensure_ascii=False)}\n\n"
        prompt = self._build_prompt(query, context, allow_asset_tags=False)
        yield from ModelScopeLLM().stream_generate([{'role': 'user', 'content': prompt}])

