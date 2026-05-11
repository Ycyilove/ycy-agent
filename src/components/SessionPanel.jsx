import { useState, useEffect } from 'react';
import { createSession, listSessions, deleteSession, renameSession } from '../services/api';

function SessionPanel({ isOpen, onClose, onSessionSelect, currentSessionId }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState('');

  useEffect(() => {
    if (isOpen) {
      loadSessions();
    }
  }, [isOpen]);

  const loadSessions = async () => {
    setLoading(true);
    try {
      const data = await listSessions();
      setSessions(data.sessions || []);
    } catch (error) {
      console.error('加载会话失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSession = async () => {
    try {
      const data = await createSession(newSessionName);
      setNewSessionName('');
      await loadSessions();
      if (data.session_id) {
        onSessionSelect(data.session_id);
      }
    } catch (error) {
      console.error('创建会话失败:', error);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    if (!confirm('确定要删除这个会话吗？')) return;
    try {
      await deleteSession(sessionId);
      await loadSessions();
      if (sessionId === currentSessionId) {
        onSessionSelect(null);
      }
    } catch (error) {
      console.error('删除会话失败:', error);
    }
  };

  const handleRenameSession = async (sessionId) => {
    if (!editName.trim()) return;
    try {
      await renameSession(sessionId, editName);
      setEditingId(null);
      setEditName('');
      await loadSessions();
    } catch (error) {
      console.error('重命名失败:', error);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl w-full max-w-md max-h-[80vh] overflow-hidden shadow-2xl border border-white/10">
        <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">会话管理</h2>
          <button onClick={onClose} className="text-white/60 hover:text-white">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-4">
          <div className="flex gap-2 mb-4">
            <input
              type="text"
              value={newSessionName}
              onChange={(e) => setNewSessionName(e.target.value)}
              placeholder="新会话名称..."
              className="flex-1 px-4 py-2 bg-white/10 border border-white/10 rounded-xl text-white placeholder-white/40 focus:outline-none focus:border-violet-500"
              onKeyPress={(e) => e.key === 'Enter' && handleCreateSession()}
            />
            <button
              onClick={handleCreateSession}
              className="px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-xl transition-colors"
            >
              新建
            </button>
          </div>

          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {loading ? (
              <div className="text-center py-8 text-white/40">加载中...</div>
            ) : sessions.length === 0 ? (
              <div className="text-center py-8 text-white/40">暂无会话</div>
            ) : (
              sessions.map((session) => (
                <div
                  key={session.session_id}
                  className={`p-3 rounded-xl transition-colors ${
                    session.session_id === currentSessionId
                      ? 'bg-violet-500/20 border border-violet-500/30'
                      : 'bg-white/5 hover:bg-white/10'
                  }`}
                >
                  {editingId === session.session_id ? (
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        className="flex-1 px-2 py-1 bg-white/10 rounded text-white text-sm"
                        autoFocus
                        onKeyPress={(e) => e.key === 'Enter' && handleRenameSession(session.session_id)}
                      />
                      <button
                        onClick={() => handleRenameSession(session.session_id)}
                        className="text-emerald-400 text-sm"
                      >
                        保存
                      </button>
                      <button
                        onClick={() => setEditingId(null)}
                        className="text-white/40 text-sm"
                      >
                        取消
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center justify-between">
                      <div className="flex-1 cursor-pointer" onClick={() => onSessionSelect(session.session_id)}>
                        <div className="text-white font-medium">{session.name || '未命名会话'}</div>
                        <div className="text-xs text-white/40">
                          {new Date(session.updated_at).toLocaleString()}
                        </div>
                      </div>
                      <div className="flex gap-1">
                        <button
                          onClick={() => { setEditingId(session.session_id); setEditName(session.name); }}
                          className="p-1.5 text-white/40 hover:text-white"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                          </svg>
                        </button>
                        <button
                          onClick={() => handleDeleteSession(session.session_id)}
                          className="p-1.5 text-white/40 hover:text-rose-400"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default SessionPanel;
