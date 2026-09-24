import { useState } from 'react';
import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { useApp } from '../../app/state/AppContext';
import { useActiveWorkspace } from '../../app/state/AppStore';

const SUPPORTS_PICKER =
  typeof window !== 'undefined' && 'showDirectoryPicker' in window;

export default function WorkspacePanel() {
  const { state, dispatch } = useApp();
  const workspace = useActiveWorkspace();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [manualMode, setManualMode] = useState(false);
  const [manualPath, setManualPath] = useState('');

  const activeSessionId = state.activeSessionId;
  const hasSession = Boolean(activeSessionId && state.sessions[activeSessionId]);

  const pickDirectory = async () => {
    setError('');
    if (!hasSession) {
      setError('请先创建或选择一个会话');
      return;
    }
    if (!SUPPORTS_PICKER) {
      setManualMode(true);
      return;
    }
    setBusy(true);
    try {
      const handle = await window.showDirectoryPicker({ mode: 'readwrite' });

      if (handle.queryPermission) {
        const perm = await handle.queryPermission({ mode: 'readwrite' });
        if (perm !== 'granted') {
          const req = await handle.requestPermission({ mode: 'readwrite' });
          if (req !== 'granted') {
            setError('未获得该目录的读写权限');
            return;
          }
        }
      }

      const exists = workspace.allowed.some((d) => d.label === handle.name);
      if (exists) {
        setError(`目录「${handle.name}」已在工作区中`);
        return;
      }

      dispatch({
        type: 'ADD_WORKSPACE_DIR',
        sessionId: activeSessionId,
        dir: {
          path: handle.name,
          label: handle.name,
          handle,
        },
      });
    } catch (err) {
      if (err?.name === 'AbortError') return;
      console.error('选择目录失败:', err);
      setError(err?.message || '选择目录失败');
    } finally {
      setBusy(false);
    }
  };

  const submitManual = () => {
    const path = manualPath.trim();
    if (!path) return;
    if (!hasSession) {
      setError('请先创建或选择一个会话');
      return;
    }
    const exists = workspace.allowed.some((d) => d.path === path);
    if (exists) {
      setError('该路径已在工作区中');
      return;
    }
    const label = path.split(/[\\/]/).filter(Boolean).pop() || path;
    dispatch({
      type: 'ADD_WORKSPACE_DIR',
      sessionId: activeSessionId,
      dir: { path, label },
    });
    setManualPath('');
    setManualMode(false);
    setError('');
  };

  const removeDir = (path) => {
    if (!hasSession) return;
    dispatch({
      type: 'REMOVE_WORKSPACE_DIR',
      sessionId: activeSessionId,
      path,
    });
  };

  return (
    <div className="border-b border-[var(--border)]">
      <div className="flex items-center justify-between px-4 pb-2 pt-4">
        <span className="text-[11px] uppercase tracking-wider text-[var(--dim)]">
          工作区
        </span>
        <button
          onClick={pickDirectory}
          disabled={busy || !hasSession}
          className="rounded p-1 text-[var(--muted)] transition-colors hover:text-[var(--text)] disabled:opacity-40"
          title="添加目录"
        >
          {busy ? (
            <svg className="h-3.5 w-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="3"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
          ) : (
            <Icon d={I.plus} className="h-3.5 w-3.5" />
          )}
        </button>
      </div>

      {!hasSession && (
        <div className="px-4 pb-3 text-[11px] text-[var(--dim)]">
          请先创建或选择一个会话
        </div>
      )}

      {manualMode && hasSession && (
        <div className="px-2 pb-2">
          <div className="flex gap-1.5">
            <input
              autoFocus
              type="text"
              value={manualPath}
              onChange={(e) => setManualPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitManual();
                if (e.key === 'Escape') {
                  setManualMode(false);
                  setManualPath('');
                  setError('');
                }
              }}
              placeholder="输入目录绝对路径…"
              className="h-7 flex-1 rounded border border-[var(--border)] bg-[var(--surface-2)] px-2 font-mono text-[11px] text-[var(--text)] placeholder-[var(--dim)] focus:border-[var(--accent)]/60 focus:outline-none"
            />
            <button
              onClick={submitManual}
              className="rounded border border-[var(--border)] px-2 text-[11px] text-[var(--muted)] transition-colors hover:border-[var(--accent)]/40 hover:text-[var(--text)]"
            >
              确定
            </button>
          </div>
          <div className="mt-1 text-[10px] text-[var(--dim)]">
            {SUPPORTS_PICKER
              ? '提示：按 Esc 取消'
              : '当前浏览器不支持文件选择器，请手动输入路径'}
          </div>
        </div>
      )}

      {error && (
        <div className="px-2 pb-2">
          <div className="rounded border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-2 py-1.5 text-[11px] text-[var(--danger)]">
            {error}
          </div>
        </div>
      )}

      {hasSession && (
        <div className="space-y-0.5 px-2 pb-3">
          {workspace.allowed.map((dir) => (
            <div
              key={dir.path}
              className="group flex items-center gap-2 rounded-md px-2.5 py-1.5 transition-colors hover:bg-[var(--surface)]"
              title={dir.handle ? `${dir.label}（由文件选择器授权）` : dir.path}
            >
              <Icon
                d={I.folder}
                className="h-3.5 w-3.5 flex-shrink-0 text-[var(--accent)]"
              />
              <span className="flex-1 truncate text-xs text-[var(--muted)]">
                {dir.label}
              </span>
              {dir.handle && (
                <span
                  className="flex-shrink-0 rounded border border-[var(--border)] px-1 text-[9px] text-[var(--dim)]"
                  title="通过文件选择器授权"
                >
                  句柄
                </span>
              )}
              <button
                onClick={() => removeDir(dir.path)}
                className="flex-shrink-0 text-[var(--dim)] opacity-0 transition-all hover:text-[var(--danger)] group-hover:opacity-100"
              >
                <Icon d={I.close} className="h-3 w-3" />
              </button>
            </div>
          ))}

          {workspace.allowed.length === 0 && !manualMode && (
            <button
              onClick={pickDirectory}
              className="w-full rounded-md border border-dashed border-[var(--border)] px-2.5 py-3 text-left text-[11px] text-[var(--dim)] transition-colors hover:border-[var(--accent)]/40 hover:text-[var(--muted)]"
            >
              点击选择文件夹作为工作区
            </button>
          )}

          {workspace.allowed.length === 0 && !manualMode && (
            <div className="px-2.5 pt-2 text-[11px] leading-relaxed text-[var(--dim)]">
              Agent 只能在工作区内直接操作文件；工作区外每次操作都需确认。
            </div>
          )}

          {SUPPORTS_PICKER && !manualMode && workspace.allowed.length > 0 && (
            <div className="px-2.5 pt-2">
              <button
                onClick={() => setManualMode(true)}
                className="text-[10px] text-[var(--dim)] transition-colors hover:text-[var(--muted)]"
              >
                手动输入路径
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}