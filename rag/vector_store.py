"""
FAISS向量存储模块
本地生成向量并存入FAISS索引，持久化保存索引文件避免重复向量化
"""
import os
# 配置 HuggingFace 国内镜像源（在导入 sentence_transformers 之前设置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 允许在线下载
import json
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np


class FAISSVectorStore:
    """FAISS向量存储"""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", index_path: str = "./vector_store"):
        """
        初始化FAISS向量存储

        Args:
            embedding_model: 嵌入模型名称
            index_path: 索引文件保存路径
        """
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

    def _init_embeddings(self):
        """初始化嵌入模型"""
        print(f"[模型下载] 正在从镜像源下载嵌入模型: {self.embedding_model}")
        print("[模型下载] 这可能需要几分钟时间，请耐心等待...")
        from sentence_transformers import SentenceTransformer
        print("[模型下载] 模型下载完成，正在加载模型...")
        self.embeddings = SentenceTransformer(self.embedding_model)
        self.embedding_dimension = self.embeddings.get_sentence_embedding_dimension()
        print("[模型下载] 模型加载成功！")

    def _load_index(self):
        """加载已存在的索引"""
        os.makedirs(self.index_path, exist_ok=True)

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

    def _get_file_hash(self, filename: str) -> str:
        """获取文件哈希值"""
        return hashlib.md5(filename.encode()).hexdigest()

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """生成文本嵌入"""
        if self.embeddings is None:
            self._init_embeddings()
        embeddings = self.embeddings.encode(texts, show_progress_bar=len(texts) > 10)
        return np.array(embeddings, dtype='float32')

    def add_documents(self, texts: List[str], filenames: List[str], metadata: List[Dict] = None) -> Dict:
        """
        添加文档到向量库

        Args:
            texts: 文本列表
            filenames: 对应的文件名列表
            metadata: 额外的元数据

        Returns:
            添加结果
        """
        if not texts or not filenames:
            return {"status": "error", "message": "文本和文件名不能为空"}

        if len(texts) != len(filenames):
            return {"status": "error", "message": "文本和文件名数量不匹配"}

        import faiss
        if self.index is None:
            if self.embeddings is None:
                self._init_embeddings()
            self.index = faiss.IndexFlatL2(self.embedding_dimension)

        added_chunks = 0
        new_file_index = {}

        for idx, (text, filename) in enumerate(zip(texts, filenames)):
            file_hash = self._get_file_hash(filename)

            if file_hash in self.file_index:
                continue

            text_list = [text] if isinstance(text, str) else text
            filename_list = [filename] if isinstance(filename, str) else filename

            embeddings = self._generate_embeddings(text_list)
            self.index.add(embeddings)

            meta = metadata[idx] if metadata and idx < len(metadata) else {}
            for i, (t, fn) in enumerate(zip(text_list, filename_list)):
                self.metadata.append({
                    'text': t,
                    'filename': fn,
                    **meta
                })

            new_file_index[file_hash] = {
                'filename': filename,
                'chunk_count': len(text_list),
                'added': True
            }
            added_chunks += len(text_list)

        self.file_index.update(new_file_index)
        self._save_index()

        return {
            "status": "success",
            "message": f"成功添加 {added_chunks} 个文本块",
            "file_count": len(new_file_index)
        }

    def _save_index(self):
        """保存索引到磁盘"""
        import faiss
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        with open(self.file_index_file, 'w', encoding='utf-8') as f:
            json.dump(self.file_index, f, ensure_ascii=False, indent=2)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            相似文档列表
        """
        if self.index is None or len(self.metadata) == 0:
            return []

        query_embedding = self._generate_embeddings([query])
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    'text': self.metadata[idx]['text'],
                    'filename': self.metadata[idx].get('filename', 'unknown'),
                    'score': float(dist),
                    'metadata': {k: v for k, v in self.metadata[idx].items() if k != 'text'}
                })

        return results

    def delete_file(self, filename: str) -> Dict:
        """
        删除指定文件的向量索引

        Args:
            filename: 要删除的文件名

        Returns:
            删除结果
        """
        if not self.file_index:
            return {"status": "error", "message": "索引为空"}

        file_hash = self._get_file_hash(filename)

        if file_hash not in self.file_index:
            return {"status": "error", "message": f"文件 {filename} 不在知识库中"}

        indices_to_remove = []
        new_metadata = []

        for idx, meta in enumerate(self.metadata):
            if meta.get('filename') == filename:
                indices_to_remove.append(idx)
            else:
                new_metadata.append(meta)

        if indices_to_remove:
            import faiss
            all_indices = list(range(self.index.ntotal))
            remaining_indices = [i for i in all_indices if i not in indices_to_remove]

            if remaining_indices:
                self.index = faiss.IndexFlatL2(self.embedding_dimension)
                vectors = self.index.reconstruct_n(0, self.index.ntotal)
                remaining_vectors = vectors[remaining_indices]
                self.index.add(remaining_vectors)
            else:
                self.index = None

            self.metadata = new_metadata
            del self.file_index[file_hash]
            self._save_index()

        return {"status": "success", "message": f"已删除文件 {filename}"}

    def clear_all(self) -> Dict:
        """
        清空全部知识库

        Returns:
            清空结果
        """
        if self.index:
            import faiss
            self.index.reset()

        self.index = None
        self.metadata = []
        self.file_index = {}

        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
        if os.path.exists(self.file_index_file):
            os.remove(self.file_index_file)

        return {"status": "success", "message": "已清空全部知识库"}

    def get_stats(self) -> Dict:
        """
        获取知识库统计信息

        Returns:
            统计信息
        """
        return {
            "total_chunks": len(self.metadata),
            "total_files": len(self.file_index),
            "files": list(self.file_index.values()),
            "embedding_model": self.embedding_model,
            "index_path": self.index_path
        }

    def file_exists(self, filename: str) -> bool:
        """检查文件是否已在知识库中"""
        file_hash = self._get_file_hash(filename)
        return file_hash in self.file_index
