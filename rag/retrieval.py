"""
检索调用模块
自动语义检索Top3相关文档片段，拼接上下文再调用大模型生成回答
"""
import json
from typing import List, Dict, Optional


class RetrievalService:
    """检索服务"""

    def __init__(self, vector_store, llm_service):
        """
        初始化检索服务

        Args:
            vector_store: FAISS向量存储实例
            llm_service: LLM服务实例
        """
        self.vector_store = vector_store
        self.llm_service = llm_service

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            相关文档列表
        """
        return self.vector_store.search(query, top_k)

    def build_context(self, retrieved_docs: List[Dict]) -> str:
        """
        构建检索上下文

        Args:
            retrieved_docs: 检索到的文档列表

        Returns:
            格式化后的上下文字符串
        """
        if not retrieved_docs:
            return "未找到相关文档"

        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc.get('filename', '未知来源')
            content = doc.get('text', '')
            score = doc.get('score', 0)

            context_parts.append(
                f"【文档 {idx}】来源: {source} (相似度: {score:.4f})\n{content}"
            )

        return "\n\n".join(context_parts)

    def generate_answer(self, query: str, context: str, history: List[Dict] = None) -> str:
        """
        生成回答

        Args:
            query: 用户问题
            context: 检索到的上下文
            history: 对话历史

        Returns:
            生成的回答
        """
        prompt = f"""基于以下参考信息回答用户问题。如果参考信息不足以回答，请基于你的知识回答，并说明情况。

参考信息:
{context}

用户问题: {query}

回答要求:
1. 优先使用参考信息回答
2. 如果参考信息不足，明确说明"根据知识库信息无法完全回答该问题"
3. 回答要准确、简洁

回答:"""

        messages = []
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        return self.llm_service.generate(messages)

    def stream_generate_answer(self, query: str, context: str, history: List[Dict] = None):
        """
        流式生成回答

        Args:
            query: 用户问题
            context: 检索到的上下文
            history: 对话历史

        Returns:
            生成器，流式输出回答
        """
        prompt = f"""基于以下参考信息回答用户问题。如果参考信息不足以回答，请基于你的知识回答，并说明情况。

参考信息:
{context}

用户问题: {query}

回答要求:
1. 优先使用参考信息回答
2. 如果参考信息不足，明确说明"根据知识库信息无法完全回答该问题"
3. 回答要准确、简洁

回答:"""

        messages = []
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        return self.llm_service.stream_generate(messages)

    def rag_answer(self, query: str, top_k: int = 3, history: List[Dict] = None) -> Dict:
        """
        RAG问答主流程

        Args:
            query: 用户问题
            top_k: 检索文档数量
            history: 对话历史

        Returns:
            包含回答和检索结果的字典
        """
        retrieved_docs = self.retrieve(query, top_k)
        context = self.build_context(retrieved_docs)
        answer = self.generate_answer(query, context, history)

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "context": context
        }

    def stream_rag_answer(self, query: str, top_k: int = 3, history: List[Dict] = None):
        """
        流式RAG问答主流程

        Args:
            query: 用户问题
            top_k: 检索文档数量
            history: 对话历史

        Returns:
            生成器，流式输出回答
        """
        retrieved_docs = self.retrieve(query, top_k)
        context = self.build_context(retrieved_docs)

        return self.stream_generate_answer(query, context, history)
