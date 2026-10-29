"""服务端权威审批状态。"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ApprovalRecord:
    approval_id: str
    task_id: str
    session_id: str
    tool_name: str
    params_hash: str
    created_at: float
    expires_at: float
    consumed: bool = False
    decision: str = "pending"


class ApprovalStore:
    def __init__(self, ttl_seconds: int = 600):
        self._records: Dict[str, ApprovalRecord] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    @staticmethod
    def _hash_params(tool_name: str, params: dict) -> str:
        payload = json.dumps(
            {"tool": tool_name, "params": params},
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def create(self, task_id, session_id, tool_name, params) -> ApprovalRecord:
        rid = uuid.uuid4().hex
        now = time.time()
        rec = ApprovalRecord(
            approval_id=rid, task_id=task_id, session_id=session_id,
            tool_name=tool_name,
            params_hash=self._hash_params(tool_name, params),
            created_at=now, expires_at=now + self._ttl,
        )
        with self._lock:
            self._records[rid] = rec
        return rec

    def decide(self, approval_id: str, session_id: str, action: str) -> bool:
        if action not in ("allow", "deny", "allow-always"):
            return False
        with self._lock:
            rec = self._records.get(approval_id)
            if rec is None or rec.consumed:
                return False
            if rec.session_id != session_id:
                return False
            if time.time() > rec.expires_at:
                return False
            rec.decision = "allow" if action != "deny" else "deny"
            rec.consumed = True
            return True

    def verify(self, approval_id: str, tool_name: str, params: dict) -> bool:
        with self._lock:
            rec = self._records.get(approval_id)
            if rec is None or not rec.consumed or rec.decision != "allow":
                return False
            if time.time() > rec.expires_at:
                return False
            return rec.params_hash == self._hash_params(tool_name, params)

    def has(self, approval_id: str) -> bool:
        """是否已存在（幂等检查用）。"""
        with self._lock:
            return approval_id in self._records
            
    def purge_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [k for k, r in self._records.items() if now > r.expires_at]
            for k in expired:
                self._records.pop(k, None)
        return len(expired)


_store: Optional[ApprovalStore] = None
_store_lock = threading.Lock()


def get_approval_store() -> ApprovalStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ApprovalStore()
    return _store