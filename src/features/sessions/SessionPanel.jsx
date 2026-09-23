import { useState, useEffect } from 'react';
import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { useApp } from '../../app/state/AppContext';
import {
  createSession,
  listSessions,
  deleteSession,
  renameSession,
} from '../../services/api';

export default function SessionPanel({ isOpen, onClose }) {
  const { state, dispatch } = useApp();
  const currentSessionId = state.activeSessionId;

  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState('');
  const [batchMode, setBatchMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState(new Set());

  useEffect(() => {
    if (isOpen) loadSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  const loadSessions = async () => {
    setLoading(true);
    try {
      const data = await listSessions();
      const list = data.sessions || [];
      setSessions(list);

      console.log('[dsh][pollution] SessionPanel 同步', list.map((s) => ({
        id: s.session_id,
        backendName: s.name,
      })));

      for (const s of list) {
        const existing = state.sessions[s.session_id];
        const willUse = existing?.name || s.name || '未命名会话';
        if (existing && s.name && existing.name !== s.name) {
          console.warn('[dsh][pollution] 后端名字会覆盖/影响本地', {
            id: s.session_id,
            localName: existing.name,
            backendName: s.name,
            willUse,
          });
        }
        dispatch({
          type: 'CREATE_SESSION',
          id: s.session_id,
          name: willUse,
        });
      }
    } catch (error) {
      console.error('[dsh][pollution] SessionPanel 加载失败', error);
    } finally {
      setLoading(false);
    }
  };

  const selectSession = (sessionId) => {
    if (!sessionId) {
      dispatch({ type: 'SET_ACTIVE_SESSION', id: null });
      return;
    }
    const target = sessions.find((s) => s.session_id === sessionId);
    if (target) {
      dispatch({
        type: 'CREATE_SESSION',
        id: target.session_id,
        name: target.name || '未命名会话',
      });
    }
    dispatch({ type: 'SET_ACTIVE_SESSION', id: sessionId });
    onClose();
  };

  const handleCreateSession = async () => {
    try {
      const data = await createSession(newSessionName);
      setNewSessionName('');
      await loadSessions();
      if (data.session_id) {
        dispatch({
          type: 'CREATE_SESSION',
          id: data.session_id,
          name: newSessionName || '未命名会话',
        });
        dispatch({ type: 'SET_ACTIVE_SESSION', id: data.session_id });
        onClose();
      }
    } catch (error) {
      console.error('创建会话失败:', error);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    if (!confirm('确定要删除这个会话吗？')) return;
    try {
      await deleteSession(sessionId);
      dispatch({ type: 'DELETE_SESSION', id: sessionId });
      await loadSessions();
      if (sessionId === currentSessionId) {
        const remaining = sessions.filter((s) => s.session_id !== sessionId);
        dispatch({
          type: 'SET_ACTIVE_SESSION',
          id: remaining[0]?.session_id || null,
        });
      }
    } catch (error) {
      console.error('删除会话失败:', error);
    }
  };

  const handleRenameSession = async (sessionId) => {
    if (!editName.trim()) return;
    try {
      await renameSession(sessionId, editName);
      dispatch({ type: 'SET_SESSION_NAME', id: sessionId, name: editName });
      setEditingId(null);
      setEditName('');
      await loadSessions();
    } catch (error) {
      console.error('重命名失败:', error);
    }
  };

  const toggleBatchMode = () => {
    setBatchMode(!batchMode);
    if (batchMode) setSelectedIds(new Set());
  };

  const toggleSelect = (sessionId) => {
    const next = new Set(selectedIds);
    if (next.has(sessionId)) next.delete(sessionId);
    else next.add(sessionId);
    setSelectedIds(next);
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === sessions.length) setSelectedIds(new Set());
    else setSelectedIds(new Set(sessions.map((s) => s.session_id)));
  };

  const handleBatchDelete = async () => {
    if (selectedIds.size === 0) return;
    if (!confirm(`确定要删除选中的 ${selectedIds.size} 个会话吗？`)) return;
    try {
      await Promise.all(Array.from(selectedIds).map((id) => deleteSession(id)));
      for (const id of selectedIds) {
        dispatch({ type: 'DELETE_SESSION', id });
      }
      const removedCurrent = selectedIds.has(currentSessionId);
      setSelectedIds(new Set());
      setBatchMode(false);
      await loadSessions();
      if (removedCurrent) {
        const remaining = sessions.filter((s) => !selectedIds.has(s.session_id));
        dispatch({
          type: 'SET_ACTIVE_SESSION',
          id: remaining[0]?.session_id || null,
        });
      }
    } catch (error) {
      console.error('批量删除会话失败:', error);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="flex max-h-[80vh] w-full max-w-md flex-col overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
        {/* Header */}
        <div className="flex flex-shrink-0 items-center justify-between border-b border-[var(--border)] px-5 py-3">
          <h2 className="text-sm font-medium text-[var(--text)]">会话管理</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={toggleBatchMode}
              className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                batchMode
                  ? 'bg-[var(--accent)] font-medium text-[var(--bg)]'
                  : 'border border-[var(--border)] text-[var(--muted)] hover:border-[var(--border-hover)] hover:text-[var(--text)]'
              }`}
            >
              {batchMode ? '取消批量' : '批量选择'}
            </button>
            <button
              onClick={onClose}
              className="rounded p-1 text-[var(--muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
            >
              <Icon d={I.close} className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex flex-1 flex-col overflow-hidden p-4">
          <div className="mb-4 flex flex-shrink-0 gap-2">
            <input
              type="text"
              value={newSessionName}
              onChange={(e) => setNewSessionName(e.target.value)}
              placeholder="新会话名称…"
              onKeyDown={(e) => e.key === 'Enter' && handleCreateSession()}
              className="h-9 flex-1 rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-3 text-sm text-[var(--text)] placeholder-[var(--dim)] transition-colors focus:border-[var(--accent)]/60 focus:outline-none"
            />
            <button
              onClick={handleCreateSession}
              className="h-9 rounded-md bg-[var(--accent)] px-3 text-sm font-medium text-[var(--bg)] transition-colors hover:bg-[var(--accent-hover)]"
            >
              新建
            </button>
          </div>

          <div className="flex-1 space-y-1.5 overflow-y-auto">
            {loading ? (
              <div className="py-8 text-center text-xs text-[var(--dim)]">
                加载中…
              </div>
            ) : sessions.length === 0 ? (
              <div className="py-8 text-center text-xs text-[var(--dim)]">
                暂无会话
              </div>
            ) : (
              sessions.map((session) => (
                <div
                  key={session.session_id}
                  className={`rounded-md border p-3 transition-colors ${
                    session.session_id === currentSessionId
                      ? 'border-[var(--accent)]/40 bg-[var(--surface-2)]'
                      : 'border-[var(--border)] bg-[var(--surface-2)] hover:border-[var(--border-hover)]'
                  }`}
                >
                  {editingId === session.session_id ? (
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        onKeyDown={(e) =>
                          e.key === 'Enter' &&
                          handleRenameSession(session.session_id)
                        }
                        autoFocus
                        className="h-7 flex-1 rounded border border-[var(--border)] bg-[var(--surface)] px-2 text-xs text-[var(--text)] focus:border-[var(--accent)]/60 focus:outline-none"
                      />
                      <button
                        onClick={() => handleRenameSession(session.session_id)}
                        className="text-xs text-[var(--accent)] hover:text-[var(--accent-hover)]"
                      >
                        保存
                      </button>
                      <button
                        onClick={() => setEditingId(null)}
                        className="text-xs text-[var(--muted)] hover:text-[var(--text)]"
                      >
                        取消
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center justify-between">
                      <div
                        className="flex min-w-0 flex-1 cursor-pointer items-center gap-2.5"
                        onClick={() => {
                          if (batchMode) toggleSelect(session.session_id);
                          else selectSession(session.session_id);
                        }}
                      >
                        {batchMode && (
                          <input
                            type="checkbox"
                            checked={selectedIds.has(session.session_id)}
                            readOnly
                            className="pointer-events-none h-3.5 w-3.5 flex-shrink-0 accent-[var(--accent)]"
                          />
                        )}
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm text-[var(--text)]">
                            {session.name || '未命名会话'}
                          </div>
                          <div className="mt-0.5 text-[11px] text-[var(--dim)]">
                            {new Date(session.updated_at).toLocaleString()}
                          </div>
                        </div>
                      </div>
                      {!batchMode && (
                        <div className="flex flex-shrink-0 gap-0.5">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditingId(session.session_id);
                              setEditName(session.name);
                            }}
                            className="rounded p-1.5 text-[var(--dim)] transition-colors hover:text-[var(--text)]"
                          >
                            <Icon d={I.edit} className="h-3.5 w-3.5" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteSession(session.session_id);
                            }}
                            className="rounded p-1.5 text-[var(--dim)] transition-colors hover:text-[var(--danger)]"
                          >
                            <Icon d={I.trash} className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>

          {batchMode && sessions.length > 0 && (
            <div className="mt-3 flex-shrink-0 border-t border-[var(--border)] pt-3">
              <div className="flex items-center justify-between">
                <label className="flex cursor-pointer items-center gap-2 text-xs text-[var(--muted)]">
                  <input
                    type="checkbox"
                    checked={
                      selectedIds.size === sessions.length && sessions.length > 0
                    }
                    onChange={toggleSelectAll}
                    className="h-3.5 w-3.5 accent-[var(--accent)]"
                  />
                  全选
                </label>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[var(--dim)]">
                    已选 {selectedIds.size} 项
                  </span>
                  <button
                    onClick={handleBatchDelete}
                    disabled={selectedIds.size === 0}
                    className={`rounded-md px-3 py-1 text-xs transition-colors ${
                      selectedIds.size > 0
                        ? 'bg-[var(--danger)] text-white hover:bg-[var(--danger-hover)]'
                        : 'cursor-not-allowed border border-[var(--border)] bg-[var(--surface-2)] text-[var(--dim)]'
                    }`}
                  >
                    批量删除
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}