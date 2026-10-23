"""
FAISS向量存储模块
本地生成向量并存入FAISS索引，持久化保存索引文件避免重复向量化
"""
import os

from infra.offline import configure_hf_env
configure_hf_env()

import json
import logging
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS向量存储"""

    def __init__(
        self,
        embedding_model: str = None,
        index_path: str = "./vector_store",
    ):
        if embedding_model is None:
            try:
                from backend.config import RAG_EMBEDDING_MODEL
                embedding_model = RAG_EMBEDDING_MODEL
            except ImportError:
                embedding_model = "BAAI/bge-small-zh-v1.5"
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "faiss.index")
        self.metadata_file = os.path.join(index_path, "metadata.pkl")
        self.file_index_file = os.path.join(index_path, "file_index.json")

        self.index = None
        self.metadata = []
        self.file_index = {}
        self.embeddings = None
        self.embedding_dimension = None

        self._load_index()

    # ────────── 内部：加载 / 保存 / 嵌入 ──────────

    def _init_embeddings(self):
        from infra.embedding import get_encoder
        self.embeddings = get_encoder(self.embedding_model, local_files_only=True)
        self.embedding_dimension = self.embeddings.get_sentence_embedding_dimension()
        logger.info(
            "[faiss] embedding ready: model=%s dim=%d",
            self.embedding_model, self.embedding_dimension,
        )

    def _load_index(self):
        real_dir = os.path.dirname(self.index_path)
        if real_dir:
            os.makedirs(real_dir, exist_ok=True)

        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            try:
                import faiss
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                if os.path.exists(self.file_index_file):
                    with open(self.file_index_file, 'r', encoding='utf-8') as f:
                        self.file_index = json.load(f)
            except Exception as e:
                print(f"加载索引失败: {e}，将创建新索引")
                self.index = None
                self.metadata = []
                self.file_index = {}

        # ── 修复：file_index.json 缺失/为空但 metadata 有内容 → 从 metadata 重建 ──
        if self.metadata and not self.file_index:
            logger.warning(
                "[faiss] file_index.json 缺失或为空，从 metadata 重建（%d 条 chunk）",
                len(self.metadata),
            )
            rebuilt: Dict[str, Dict] = {}
            for meta in self.metadata:
                fn = meta.get("filename")
                if not fn:
                    continue
                fh = self._get_file_hash(fn)
                if fh not in rebuilt:
                    rebuilt[fh] = {
                        "filename": fn,
                        "chunk_count": 0,
                        "added": True,
                    }
                rebuilt[fh]["chunk_count"] += 1
            self.file_index = rebuilt
            try:
                with open(self.file_index_file, 'w', encoding='utf-8') as f:
                    json.dump(self.file_index, f, ensure_ascii=False, indent=2)
                logger.info("[faiss] file_index 重建完成，%d 个文件", len(rebuilt))
            except Exception:
                logger.exception("[faiss] file_index 写回失败")

        # ── 修复：索引维度与模型维度不一致 → 标记索引失效，等首次加载模型时校验 ──
        if self.index is not None and self.metadata:
            index_dim = self.index.d
            logger.info(
                "[faiss] 索引加载：ntotal=%d, dim=%d, metadata=%d 条",
                self.index.ntotal, index_dim, len(self.metadata),
            )

    def _get_file_hash(self, filename: str) -> str:
        return hashlib.md5(filename.encode()).hexdigest()

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.embeddings is None:
            self._init_embeddings()
        embeddings = self.embeddings.encode(texts, show_progress_bar=len(texts) > 10)
        return np.array(embeddings, dtype='float32')

    # ────────── 一致性自检 ──────────

    def verify_consistency(self) -> Dict:
        """检查 index 和 metadata 是否一致。"""
        index_count = self.index.ntotal if self.index is not None else 0
        meta_count = len(self.metadata)
        return {
            "index_ntotal": index_count,
            "metadata_len": meta_count,
            "consistent": index_count == meta_count,
            "file_index_len": len(self.file_index),
            "index_dim": self.index.d if self.index is not None else None,
        }

    def _check_dim(self, query_dim: int) -> bool:
        """检查 query 向量维度与索引维度是否一致。"""
        if self.index is None:
            return False
        if self.index.d != query_dim:
            logger.error(
                "[faiss] 维度不匹配：index.d=%d, query.d=%d。"
                "embedding 模型已更换（%s），旧索引失效，请清空 vector_store/ 重建。",
                self.index.d, query_dim, self.embedding_model,
            )
            return False
        return True

    # ────────── 写入 ──────────

    def add_documents(
        self,
        texts: List[str],
        filenames: List[str],
        metadata: List[Dict] = None,
    ) -> Dict:
        if not texts or not filenames:
            return {"status": "error", "message": "文本和文件名不能为空"}
        if len(texts) != len(filenames):
            return {"status": "error", "message": "文本和文件名数量不匹配"}

        import faiss
        if self.index is None:
            if self.embeddings is None:
                self._init_embeddings()
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        else:
            # 已有索引但维度不一致 → 拒绝写入，提示重建
            if self.embeddings is None:
                self._init_embeddings()
            if self.index.d != self.embedding_dimension:
                msg = (
                    f"索引维度 {self.index.d} 与当前模型 {self.embedding_dimension} 不一致，"
                    f"请先清空 vector_store/ 重建索引"
                )
                logger.error("[faiss] %s", msg)
                return {"status": "error", "message": msg}

        added_chunks = 0
        new_file_index = {}

        grouped: Dict[str, List[tuple]] = {}
        for i, (text, filename) in enumerate(zip(texts, filenames)):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            grouped.setdefault(filename, []).append((text, meta))

        for filename, items in grouped.items():
            file_hash = self._get_file_hash(filename)
            if file_hash in self.file_index:
                continue

            chunk_texts = [t for t, _ in items]
            if not chunk_texts:
                continue

            try:
                embeddings = self._generate_embeddings(chunk_texts)
                self.index.add(embeddings)
                for t, meta in items:
                    self.metadata.append({
                        'text': t,
                        'filename': filename,
                        **meta,
                    })
                new_file_index[file_hash] = {
                    'filename': filename,
                    'chunk_count': len(chunk_texts),
                    'added': True,
                }
                added_chunks += len(chunk_texts)
            except Exception:
                logger.exception("添加文件 %s 失败，回滚索引", filename)
                self._load_index()
                raise

        self.file_index.update(new_file_index)
        self._save_index()

        return {
            "status": "success",
            "message": f"成功添加 {added_chunks} 个文本块",
            "file_count": len(new_file_index),
        }

    def _save_index(self):
        import faiss
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        with open(self.file_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.file_index, f, ensure_ascii=False, indent=2)

    # ────────── 检索 ──────────

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index is None or len(self.metadata) == 0:
            return []

        query_embedding = self._generate_embeddings([query])

        # ── 修复：维度不匹配时不再 AssertionError，返回空 + 日志 ──
        if not self._check_dim(query_embedding.shape[1]):
            return []

        distances, indices = self.index.search(
            query_embedding, min(top_k, len(self.metadata))
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    'text': self.metadata[idx]['text'],
                    'filename': self.metadata[idx].get('filename', 'unknown'),
                    'score': float(dist),
                    'metadata': {
                        k: v for k, v in self.metadata[idx].items() if k != 'text'
                    },
                })
        return results

    # ────────── 删除 / 清空 ──────────

    def delete_file(self, filename: str) -> Dict:
        if not self.file_index:
            return {"status": "error", "message": "索引为空"}

        file_hash = self._get_file_hash(filename)
        if file_hash not in self.file_index:
            return {"status": "error", "message": f"文件 {filename} 不在知识库中"}

        import faiss

        indices_to_remove = [
            i for i, meta in enumerate(self.metadata)
            if meta.get('filename') == filename
        ]

        remaining_metadata = [
            meta for i, meta in enumerate(self.metadata)
            if i not in indices_to_remove
        ]

        try:
            if self.index is not None and self.index.ntotal > 0:
                vectors = self.index.reconstruct_n(0, self.index.ntotal)
                keep_indices = [
                    i for i in range(self.index.ntotal)
                    if i not in indices_to_remove
                ]
                if keep_indices:
                    remaining_vectors = vectors[keep_indices]
                    dim = remaining_vectors.shape[1]
                    new_index = faiss.IndexFlatL2(dim)
                    new_index.add(remaining_vectors)
                    self.index = new_index
                else:
                    self.index = None
            else:
                self.index = None
        except Exception as e:
            logger.warning("reconstruct 失败，降级为清空索引: %s", e)
            self.index = None

        self.metadata = remaining_metadata
        del self.file_index[file_hash]
        self._save_index()
        return {"status": "success", "message": f"已删除文件 {filename}"}

    def clear_all(self) -> Dict:
        if self.index:
            import faiss
            self.index.reset()

        self.index = None
        self.metadata = []
        self.file_index = {}

        for path in (self.index_file, self.metadata_file, self.file_index_file):
            if os.path.exists(path):
                os.remove(path)

        return {"status": "success", "message": "已清空全部知识库"}

    # ────────── 统计 / 查询 ──────────

    def get_stats(self) -> Dict:
        return {
            "total_chunks": len(self.metadata),
            "total_files": len(self.file_index),
            "files": list(self.file_index.values()),
            "embedding_model": self.embedding_model,
            "index_path": self.index_path,
            "index_dim": self.index.d if self.index is not None else None,
        }

    def file_exists(self, filename: str) -> bool:
        file_hash = self._get_file_hash(filename)
        return file_hash in self.file_index