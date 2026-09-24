"""
轻量化持久化会话记忆系统
为Python AI Agent提供会话历史管理和RAG文件隔离
基于SQLite，无需额外安装服务
"""
import sqlite3
import uuid
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
from threading import Lock


class SessionMemory:
    """
    会话记忆系统核心类
    支持会话隔离、消息持久化、文件关联、任务持久化
    """

    def __init__(self, db_path: str = "./data/sessions.db"):
        self.db_path = db_path
        self._lock = Lock()
        self._ensure_db_dir()
        self._init_tables()

    def _ensure_db_dir(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        name TEXT DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        is_active INTEGER DEFAULT 1
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        file_ids TEXT DEFAULT '[]',
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_files (
                        file_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        file_path TEXT,
                        file_type TEXT,
                        uploaded_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        title TEXT DEFAULT '',
                        status TEXT DEFAULT 'running',
                        mode TEXT DEFAULT 'plan',
                        workspace TEXT DEFAULT '[]',
                        skills TEXT DEFAULT '[]',
                        attachments TEXT DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS task_steps (
                        step_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        kind TEXT DEFAULT 'think',
                        label TEXT DEFAULT '',
                        status TEXT DEFAULT 'pending',
                        path TEXT,
                        log TEXT,
                        approval TEXT,
                        order_index INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, timestamp)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_files_session
                    ON session_files(session_id)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tasks_session
                    ON tasks(session_id, created_at)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_steps_task
                    ON task_steps(task_id, order_index)
                """)

                conn.commit()

    # ────────── 会话 ──────────

    def create_session(self, name: str = "") -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO sessions (session_id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, name, now, now)
                )
                conn.commit()
        return session_id

    def ensure_session(self, session_id: str, name: str = "") -> str:
        if not session_id:
            return self.create_session(name)
        now = datetime.now().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                if cursor.fetchone():
                    return session_id
                cursor.execute(
                    "INSERT INTO sessions (session_id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, name, now, now)
                )
                conn.commit()
        return session_id

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
                if not cursor.fetchone():
                    return False
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                return True

    def clear_all_sessions(self) -> int:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sessions")
                count = cursor.fetchone()[0]
                cursor.execute("DELETE FROM sessions")
                conn.commit()
                return count

    def rename_session(self, session_id: str, new_name: str) -> bool:
        now = datetime.now().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE sessions SET name = ?, updated_at = ? WHERE session_id = ?",
                    (new_name, now, session_id)
                )
                conn.commit()
                return cursor.rowcount > 0

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id, name, created_at, updated_at FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "session_id": row["session_id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id, name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
            rows = cursor.fetchall()
            return [
                {
                    "session_id": row["session_id"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                for row in rows
            ]

    # ────────── 消息 ──────────

    def add_message(self, session_id, role, content, file_ids=None):
        timestamp = datetime.now().isoformat()
        file_ids_json = json.dumps(file_ids or [])
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO messages (session_id, role, content, timestamp, file_ids) VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, timestamp, file_ids_json)
                )
                cursor.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (timestamp, session_id)
                )
                conn.commit()
                return cursor.lastrowid

    def get_messages(self, session_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, role, content, timestamp, file_ids FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "file_ids": json.loads(row["file_ids"])
                }
                for row in rows
            ]

    def get_conversation_context(self, session_id, max_messages=20):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (session_id, max_messages)
            )
            rows = cursor.fetchall()
            messages = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
            return messages

    def search_messages(self, session_id, keyword, limit=10):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, role, content, timestamp, file_ids
                   FROM messages
                   WHERE session_id = ? AND content LIKE ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (session_id, f"%{keyword}%", limit)
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "file_ids": json.loads(row["file_ids"])
                }
                for row in rows
            ]

    # ────────── 文件 ──────────

    def add_session_file(self, session_id, file_name, file_path=None, file_type=None):
        file_id = str(uuid.uuid4())
        uploaded_at = datetime.now().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO session_files (file_id, session_id, file_name, file_path, file_type, uploaded_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (file_id, session_id, file_name, file_path, file_type, uploaded_at)
                )
                conn.commit()
        return file_id

    def get_session_files(self, session_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_id, file_name, file_path, file_type, uploaded_at FROM session_files WHERE session_id = ? ORDER BY uploaded_at DESC",
                (session_id,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "file_id": row["file_id"],
                    "file_name": row["file_name"],
                    "file_path": row["file_path"],
                    "file_type": row["file_type"],
                    "uploaded_at": row["uploaded_at"]
                }
                for row in rows
            ]

    def delete_session_file(self, file_id):
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM session_files WHERE file_id = ?", (file_id,))
                conn.commit()
                return cursor.rowcount > 0

    def get_file_by_id(self, file_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_id, session_id, file_name, file_path, file_type, uploaded_at FROM session_files WHERE file_id = ?",
                (file_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "file_id": row["file_id"],
                "session_id": row["session_id"],
                "file_name": row["file_name"],
                "file_path": row["file_path"],
                "file_type": row["file_type"],
                "uploaded_at": row["uploaded_at"]
            }

    # ────────── 任务 ──────────

    def create_task(self, task: Dict[str, Any]) -> bool:
        """创建任务。幂等：已存在则不覆盖。"""
        now = datetime.now().isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT task_id FROM tasks WHERE task_id = ?",
                    (task["task_id"],)
                )
                if cursor.fetchone():
                    return False
                cursor.execute(
                    """INSERT INTO tasks
                       (task_id, session_id, title, status, mode, workspace, skills, attachments, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        task["task_id"],
                        task["session_id"],
                        task.get("title", ""),
                        task.get("status", "running"),
                        task.get("mode", "plan"),
                        json.dumps(task.get("workspace", []), ensure_ascii=False),
                        json.dumps(task.get("skills", []), ensure_ascii=False),
                        json.dumps(task.get("attachments", []), ensure_ascii=False),
                        now,
                        now,
                    )
                )
                conn.commit()
                return True

    def update_task(self, task_id: str, **patch) -> bool:
        if not patch:
            return False
        now = datetime.now().isoformat()
        allowed = {"title", "status", "mode", "workspace", "skills", "attachments"}
        fields = []
        values = []
        for key, value in patch.items():
            if key not in allowed:
                continue
            fields.append(f"{key} = ?")
            if key in {"workspace", "skills", "attachments"}:
                values.append(json.dumps(value, ensure_ascii=False))
            else:
                values.append(value)
        if not fields:
            return False
        fields.append("updated_at = ?")
        values.append(now)
        values.append(task_id)
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"UPDATE tasks SET {', '.join(fields)} WHERE task_id = ?",
                    values
                )
                conn.commit()
                return cursor.rowcount > 0

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                return None
            task = self._task_row_to_dict(row)
            cursor.execute(
                "SELECT * FROM task_steps WHERE task_id = ? ORDER BY order_index ASC",
                (task_id,)
            )
            task["steps"] = [self._step_row_to_dict(r) for r in cursor.fetchall()]
            return task

    def list_tasks(self, session_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM tasks WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,)
            )
            task_rows = cursor.fetchall()
            tasks = []
            for row in task_rows:
                task = self._task_row_to_dict(row)
                cursor.execute(
                    "SELECT * FROM task_steps WHERE task_id = ? ORDER BY order_index ASC",
                    (task["id"],)
                )
                task["steps"] = [self._step_row_to_dict(r) for r in cursor.fetchall()]
                tasks.append(task)
            return tasks

    def delete_task(self, task_id: str) -> bool:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
                conn.commit()
                return cursor.rowcount > 0

    def upsert_step(self, task_id: str, step: Dict[str, Any]) -> bool:
        """插入或更新步骤。已存在则合并字段。"""
        now = datetime.now().isoformat()
        step_id = step.get("id")
        if not step_id:
            return False
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT step_id FROM task_steps WHERE step_id = ?",
                    (step_id,)
                )
                exists = cursor.fetchone() is not None

                if exists:
                    fields = []
                    values = []
                    for key in ("kind", "label", "status", "path", "log"):
                        if key in step:
                            fields.append(f"{key} = ?")
                            values.append(step[key])
                    if "approval" in step:
                        fields.append("approval = ?")
                        values.append(
                            json.dumps(step["approval"], ensure_ascii=False)
                            if step["approval"] is not None else None
                        )
                    if not fields:
                        return False
                    fields.append("updated_at = ?")
                    values.append(now)
                    values.append(step_id)
                    cursor.execute(
                        f"UPDATE task_steps SET {', '.join(fields)} WHERE step_id = ?",
                        values
                    )
                else:
                    cursor.execute(
                        "SELECT COALESCE(MAX(order_index), -1) + 1 FROM task_steps WHERE task_id = ?",
                        (task_id,)
                    )
                    order_index = cursor.fetchone()[0]
                    cursor.execute(
                        """INSERT INTO task_steps
                           (step_id, task_id, kind, label, status, path, log, approval, order_index, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            step_id,
                            task_id,
                            step.get("kind", "think"),
                            step.get("label", ""),
                            step.get("status", "pending"),
                            step.get("path"),
                            step.get("log"),
                            json.dumps(step["approval"], ensure_ascii=False)
                            if step.get("approval") is not None else None,
                            order_index,
                            now,
                            now,
                        )
                    )
                conn.commit()
                return True

    # ────────── 内部工具 ──────────

    @staticmethod
    def _task_row_to_dict(row) -> Dict[str, Any]:
        return {
            "id": row["task_id"],
            "task_id": row["task_id"],
            "sessionId": row["session_id"],
            "session_id": row["session_id"],
            "title": row["title"],
            "status": row["status"],
            "mode": row["mode"],
            "workspace": json.loads(row["workspace"] or "[]"),
            "skills": json.loads(row["skills"] or "[]"),
            "attachments": json.loads(row["attachments"] or "[]"),
            "createdAt": row["created_at"],
            "created_at": row["created_at"],
            "updatedAt": row["updated_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _step_row_to_dict(row) -> Dict[str, Any]:
        return {
            "id": row["step_id"],
            "step_id": row["step_id"],
            "taskId": row["task_id"],
            "task_id": row["task_id"],
            "kind": row["kind"],
            "label": row["label"],
            "status": row["status"],
            "path": row["path"],
            "log": row["log"],
            "approval": json.loads(row["approval"]) if row["approval"] else None,
            "orderIndex": row["order_index"],
        }


# 全局单例
_memory_instance: Optional[SessionMemory] = None
_memory_lock = Lock()


def get_session_memory(db_path: str = "./data/sessions.db") -> SessionMemory:
    global _memory_instance
    if _memory_instance is None:
        with _memory_lock:
            if _memory_instance is None:
                _memory_instance = SessionMemory(db_path)
    return _memory_instance


def init_session_memory(db_path: str = "./data/sessions.db") -> SessionMemory:
    global _memory_instance
    with _memory_lock:
        _memory_instance = SessionMemory(db_path)
    return _memory_instance