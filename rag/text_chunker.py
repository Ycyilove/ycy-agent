"""
文本分块模块
实现固定块长+重叠切片的智能分块，保证语义完整性
"""
import re
from typing import List, Tuple


class TextChunker:
    """文本分块器"""

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        初始化分块器

        Args:
            chunk_size: 每个文本块的最大字符数
            overlap: 相邻块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        对文本进行分块

        Args:
            text: 待分块的文本

        Returns:
            文本块列表
        """
        if not text or not text.strip():
            return []

        text = self._preprocess_text(text)
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            if end >= text_length:
                chunks.append(text[start:])
                break

            chunk = text[start:end]

            chunk = self._split_at_sentence_boundary(chunk)

            if chunk.strip():
                chunks.append(chunk)

            start = start + len(chunk) - self.overlap
            if start < 0:
                start = 0

        return [chunk for chunk in chunks if chunk.strip()]

    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        text = re.sub(r'[\r\n]+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _split_at_sentence_boundary(self, chunk: str) -> str:
        """在句子边界处分割，保证语义完整性"""
        if len(chunk) < self.chunk_size * 0.8:
            return chunk

        sentence_endings = r'[。！？\.!?\n]+'
        matches = list(re.finditer(sentence_endings, chunk))

        if not matches:
            return chunk

        last_ending_pos = matches[-1].end()

        if last_ending_pos > len(chunk) * 0.6:
            return chunk[:last_ending_pos]

        return chunk

    def chunk_documents(self, documents: List[str], metadata: List[dict] = None) -> List[Tuple[str, dict]]:
        """
        对多个文档进行分块

        Args:
            documents: 文档列表
            metadata: 每个文档的元数据列表

        Returns:
            (文本块, 元数据)元组列表
        """
        chunks_with_metadata = []

        for idx, doc in enumerate(documents):
            doc_chunks = self.chunk_text(doc)
            doc_meta = metadata[idx] if metadata and idx < len(metadata) else {}

            for chunk_idx, chunk in enumerate(doc_chunks):
                chunk_meta = {**doc_meta, 'chunk_index': chunk_idx, 'source_doc': idx}
                chunks_with_metadata.append((chunk, chunk_meta))

        return chunks_with_metadata
