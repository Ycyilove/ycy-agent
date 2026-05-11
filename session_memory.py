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
    支持会话隔离、消息持久化、文件关联、分级管理
    """

    def __init__(self, db_path: str = "./data/sessions.db"):
        """
        初始化会话记忆系统

        Args:
            db_path: SQLite数据库文件路径（支持相对/绝对路径）
        """
        self.db_path = db_path
        self._lock = Lock()
        self._ensure_db_dir()
        self._init_tables()

    def _ensure_db_dir(self):
        """确保数据库目录存在"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """获取数据库连接（线程安全）"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        """初始化数据库表（自动建表容错）"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 会话表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        name TEXT DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        is_active INTEGER DEFAULT 1
                    )
                """)

                # 消息表
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

                # 文件关联表
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

                # 索引优化
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, timestamp)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_files_session
                    ON session_files(session_id)
                """)

                conn.commit()

    def create_session(self, name: str = "") -> str:
        """
        创建新会话

        Args:
            name: 会话名称/备注

        Returns:
            session_id: 生成的唯一会话ID
        """
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

    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话（级联删除消息和文件）

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
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
        """
        清空全部会话

        Returns:
            删除的会话数量
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sessions")
                count = cursor.fetchone()[0]

                cursor.execute("DELETE FROM sessions")
                conn.commit()
                return count

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """
        重命名会话备注

        Args:
            session_id: 会话ID
            new_name: 新名称

        Returns:
            是否重命名成功
        """
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
        """
        获取会话基本信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息字典，不存在返回None
        """
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
        """
        列出所有会话（按更新时间倒序）

        Returns:
            会话信息列表
        """
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

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        file_ids: List[str] = None
    ) -> int:
        """
        添加消息到会话

        Args:
            session_id: 会话ID
            role: 角色 (user/assistant/system)
            content: 消息内容
            file_ids: 关联的文件ID列表

        Returns:
            消息ID
        """
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

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取会话全部消息

        Args:
            session_id: 会话ID

        Returns:
            消息列表（按时间正序）
        """
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

    def get_conversation_context(
        self,
        session_id: str,
        max_messages: int = 20
    ) -> List[Dict[str, str]]:
        """
        获取短期上下文（用于实时对话拼接）

        Args:
            session_id: 会话ID
            max_messages: 最大消息数

        Returns:
            消息列表（适合直接传给LLM的格式）
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (session_id, max_messages)
            )
            rows = cursor.fetchall()
            # 反转顺序，保持时间正序
            messages = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
            return messages

    def add_session_file(
        self,
        session_id: str,
        file_name: str,
        file_path: str = None,
        file_type: str = None
    ) -> str:
        """
        添加文件到会话（实现跨会话文件隔离）

        Args:
            session_id: 会话ID
            file_name: 文件名
            file_path: 文件路径（可选）
            file_type: 文件类型（可选）

        Returns:
            file_id: 生成的文件ID
        """
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

    def get_session_files(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取会话关联的全部文件

        Args:
            session_id: 会话ID

        Returns:
            文件信息列表
        """
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

    def delete_session_file(self, file_id: str) -> bool:
        """
        删除会话文件

        Args:
            file_id: 文件ID

        Returns:
            是否删除成功
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM session_files WHERE file_id = ?", (file_id,))
                conn.commit()
                return cursor.rowcount > 0

    def get_file_by_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        根据文件ID获取文件信息

        Args:
            file_id: 文件ID

        Returns:
            文件信息，不存在返回None
        """
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

    def search_messages(
        self,
        session_id: str,
        keyword: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索会话中的消息

        Args:
            session_id: 会话ID
            keyword: 搜索关键词
            limit: 返回结果数量

        Returns:
            匹配的消息列表
        """
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


# 全局单例
_memory_instance: Optional[SessionMemory] = None
_memory_lock = Lock()


def get_session_memory(db_path: str = "./data/sessions.db") -> SessionMemory:
    """
    获取会话记忆系统单例

    Args:
        db_path: 数据库路径

    Returns:
        SessionMemory实例
    """
    global _memory_instance
    if _memory_instance is None:
        with _memory_lock:
            if _memory_instance is None:
                _memory_instance = SessionMemory(db_path)
    return _memory_instance


def init_session_memory(db_path: str = "./data/sessions.db") -> SessionMemory:
    """
    显式初始化会话记忆系统（可选）

    Args:
        db_path: 数据库路径

    Returns:
        SessionMemory实例
    """
    global _memory_instance
    with _memory_lock:
        _memory_instance = SessionMemory(db_path)
    return _memory_instance
