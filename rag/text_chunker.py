"""
文本分块模块
结构化递归分块：按 段落 → 行 → 句子 → 子句 → 字符 的优先级切分，
保证语义边界不被破坏，相邻块保留重叠。
"""

import re
from typing import List, Optional, Tuple


# ── 通用分隔符优先级（不针对特定文档类型）──
# 顺序即优先级：越靠前越优先用于切分。
# 同时兼容中英文标点。
DEFAULT_SEPARATORS = [
    "\n\n",       # 空行（段落边界）
    "\n",         # 换行（行边界）
    "\u3002",     # 。中文句号
    "\uff01",     # ！中文感叹号
    "\uff1f",     # ？中文问号
    ".",          # .  英文句号
    "!",          # !  英文感叹号
    "?",          # ?  英文问号
    "\uff1b",     # ；中文分号
    ";",          # ;  英文分号
    "\uff0c",     # ，中文逗号
    ",",          # ,  英文逗号
    "\u3001",     # 、中文顿号
    " ",          # 空格
    "",           # 字符兜底
]


class TextChunker:
    """结构化递归文本分块器。"""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ):
        """
        Args:
            chunk_size: 每个文本块的目标最大字符数
            overlap: 相邻块之间的重叠字符数
            separators: 自定义分隔符优先级列表；None 时用 DEFAULT_SEPARATORS
            keep_separator: 切分时是否保留分隔符（保留标点有助于语义完整）
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if overlap < 0:
            raise ValueError("overlap 不能为负")
        if overlap >= chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self.keep_separator = keep_separator

    # ────────── 对外入口 ──────────

    def chunk_text(self, text: str) -> List[str]:
        """把文本切成带重叠的块。"""
        if not text or not text.strip():
            return []

        text = self._preprocess_text(text)
        if not text:
            return []

        # 顶层按段落切，逐段递归
        chunks: List[str] = []
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.chunk_size:
                chunks.append(para)
            else:
                chunks.extend(self._split_recursive(para, 0))

        chunks = [c for c in chunks if c and c.strip()]
        return self._apply_overlap(chunks)

    def chunk_documents(
        self,
        documents: List[str],
        metadata: Optional[List[dict]] = None,
    ) -> List[Tuple[str, dict]]:
        """对多个文档分块，保留元数据。"""
        chunks_with_metadata: List[Tuple[str, dict]] = []
        for idx, doc in enumerate(documents):
            doc_chunks = self.chunk_text(doc)
            doc_meta = metadata[idx] if metadata and idx < len(metadata) else {}
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunk_meta = {
                    **doc_meta,
                    "chunk_index": chunk_idx,
                    "source_doc": idx,
                }
                chunks_with_metadata.append((chunk, chunk_meta))
        return chunks_with_metadata

    # ────────── 内部：预处理 ──────────

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """统一换行、压缩多余空白，但保留段落结构。"""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ────────── 内部：递归切分 ──────────

    def _split_recursive(self, text: str, depth: int) -> List[str]:
        """按 separators[depth] 切分；不够再往下一级递归。"""
        if len(text) <= self.chunk_size:
            return [text]

        if depth >= len(self.separators):
            return self._hard_split(text)

        sep = self.separators[depth]

        # 空字符串兜底：字符级硬切
        if sep == "":
            return self._hard_split(text)

        if sep not in text:
            # 当前分隔符不出现，换下一级
            return self._split_recursive(text, depth + 1)

        pieces = self._split_keep_separator(text, sep)
        return self._merge_pieces(pieces, sep, depth)

    @staticmethod
    def _split_keep_separator(text: str, sep: str) -> List[str]:
        """按 sep 切分，并把 sep 保留在每段末尾。"""
        parts = text.split(sep)
        pieces = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                pieces.append(part + sep)
            else:
                if part:
                    pieces.append(part)
        return pieces

    def _merge_pieces(
        self,
        pieces: List[str],
        sep: str,
        depth: int,
    ) -> List[str]:
        """把 pieces 按 chunk_size 合并；超长的 piece 递归下钻。"""
        chunks: List[str] = []
        buffer = ""

        for piece in pieces:
            if not piece:
                continue

            # 单个 piece 就超长 → 递归下一级
            if len(piece) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._split_recursive(piece, depth + 1))
                continue

            candidate = buffer + piece
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = piece

        if buffer:
            chunks.append(buffer)

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        """字符级硬切（最终兜底），带前进保护。"""
        chunks: List[str] = []
        start = 0
        prev_start = -1
        step = max(1, self.chunk_size - self.overlap)

        while start < len(text):
            if start <= prev_start:
                start = prev_start + 1
                if start >= len(text):
                    break
            prev_start = start

            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            chunks.append(text[start:end])
            start += step

        return chunks

    # ────────── 内部：重叠 ──────────

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """在相邻块之间加 overlap，且不破坏块内语义。"""
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = result[-1]
            tail = prev[-self.overlap:] if len(prev) > self.overlap else prev
            merged = tail + chunks[i]
            # 防止 overlap 导致块无限膨胀
            if len(merged) > self.chunk_size * 1.5:
                result.append(chunks[i])
            else:
                result.append(merged)
        return result