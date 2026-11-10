"""生命周期服务：delete_file / clear_all / get_stats。

B14-3 从 rag/service.py 原样搬（delete_file 8 + clear_all 8 + get_stats 5）。
LifecycleService.__init__ 做引用转发（self.xxx = rag.xxx）。
"""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class LifecycleService:
    """生命周期服务。self.xxx 是 rag.xxx 的引用转发。"""

    def __init__(self, rag):
        self.rag = rag
        self.vector_store = rag.vector_store
        self.bm25_store = rag.bm25_store
        self.resource_manager = rag.resource_manager
        self.record_store = rag.record_store
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
