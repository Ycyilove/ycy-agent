"""测试用 fake 实现（不依赖真实 DB / 网络 / LLM）。"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional


class FakeSession:
    """AgentTaskService 需要的 11 个方法的纯内存实现。

    线程安全（stream_task 在 thread pool 里跑，add_message 可能并发）。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.messages: Dict[str, List[Dict[str, Any]]] = {}
        self.session_files: Dict[str, List[Dict[str, Any]]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.steps: Dict[str, List[Dict[str, Any]]] = {}
        self.file_logs: Dict[str, List[Dict[str, Any]]] = {}

    # ── session ──
    def ensure_session(self, session_id: str, name: str = "") -> str:
        with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "session_id": session_id, "name": name or "会话",
                }
            return session_id

    def create_session(self, name: str = "") -> str:
        sid = f"s-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self.sessions[sid] = {"session_id": sid, "name": name or "会话"}
        return sid

    def get_conversation_context(
        self, session_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        with self._lock:
            msgs = self.messages.get(session_id, [])
            return [dict(m) for m in msgs[-limit:]]

    # ── message ──
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        file_ids: Optional[List[str]] = None,
    ) -> None:
        with self._lock:
            self.messages.setdefault(session_id, []).append({
                "role": role,
                "content": content,
                "file_ids": list(file_ids or []),
            })

    # ── session file ──
    def add_session_file(
        self, session_id: str, file_name: str, file_type: str = ""
    ) -> str:
        fid = f"f-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self.session_files.setdefault(session_id, []).append({
                "file_id": fid, "file_name": file_name, "file_type": file_type,
            })
        return fid

    def get_session_files(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(f) for f in self.session_files.get(session_id, [])]

    # ── task ──
    def create_task(self, task: Dict[str, Any]) -> None:
        with self._lock:
            self.tasks[task["task_id"]] = dict(task)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            t = self.tasks.get(task_id)
            return dict(t) if t else None

    def update_task(self, task_id: str, **kwargs: Any) -> None:
        with self._lock:
            t = self.tasks.get(task_id)
            if t is not None:
                t.update(kwargs)

    def upsert_step(self, task_id: str, step: Dict[str, Any]) -> None:
        with self._lock:
            steps = self.steps.setdefault(task_id, [])
            sid = step.get("id")
            for i, s in enumerate(steps):
                if s.get("id") == sid:
                    steps[i] = dict(step)
                    return
            steps.append(dict(step))

    # ── file log ──
    def add_file_log(self, session_id: str, entry: Dict[str, Any]) -> None:
        with self._lock:
            self.file_logs.setdefault(session_id, []).append(dict(entry))