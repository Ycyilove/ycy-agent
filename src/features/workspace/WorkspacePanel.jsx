import { useState } from 'react';
import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { useApp } from '../../app/state/AppContext';
import { useActiveWorkspace } from '../../app/state/AppStore';

const ABSOLUTE_PATH_RE = /^([A-Za-z]:[\\/]|\/)/;

export default function WorkspacePanel() {
  const { state, dispatch } = useApp();
  const workspace = useActiveWorkspace();
  const [manualMode, setManualMode] = useState(false);
  const [manualPath, setManualPath] = useState('');
  const [error, setError] = useState('');

  const activeSessionId = state.activeSessionId;
  const hasSession = Boolean(activeSessionId && state.sessions[activeSessionId]);

  const submitManual = () => {
    const raw = manualPath.trim();
    if (!raw) return;
    if (!hasSession) {
      setError('请先创建或选择一个会话');
      return;
    }
    if (!ABSOLUTE_PATH_RE.test(raw)) {
      setError('请输入绝对路径（如 D:/projects/foo 或 /home/user/foo）');
      return;
    }
    const normalized = raw.replace(/\\/g, '/').replace(/\/+$/, '');
    if (workspace.allowed.some((d) => d.path === normalized)) {
      setError('该路径已在工作区中');
      return;
    }
    const label = normalized.split('/').filter(Boolean).pop() || normalized;
    dispatch({
      type: 'ADD_WORKSPACE_DIR',
      sessionId: activeSessionId,
      dir: { path: normalized, label },
    });
    setManualPath('');
    setManualMode(false);
    setError('');
  };

  const cancelManual = () => {
    setManualMode(false);
    setManualPath('');
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
          onClick={() => setManualMode(true)}
          disabled={!hasSession || manualMode}
          className="rounded p-1 text-[var(--muted)] transition-colors hover:text-[var(--text)] disabled:opacity-40"
          title="添加目录（绝对路径）"
        >
          <Icon d={I.plus} className="h-3.5 w-3.5" />
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
                if (e.key === 'Escape') cancelManual();
              }}
              placeholder="输入目录绝对路径，如 D:/projects/foo"
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
            浏览器不暴露文件选择器的真实路径，请手动输入绝对路径（Esc 取消）
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
              title={dir.path}
            >
              <Icon
                d={I.folder}
                className="h-3.5 w-3.5 flex-shrink-0 text-[var(--accent)]"
              />
              <span className="flex-1 truncate text-xs text-[var(--muted)]">
                {dir.label}
              </span>
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
              onClick={() => setManualMode(true)}
              className="w-full rounded-md border border-dashed border-[var(--border)] px-2.5 py-3 text-left text-[11px] text-[var(--dim)] transition-colors hover:border-[var(--accent)]/40 hover:text-[var(--muted)]"
            >
              点击输入目录绝对路径作为工作区
            </button>
          )}

          {workspace.allowed.length === 0 && !manualMode && (
            <div className="px-2.5 pt-2 text-[11px] leading-relaxed text-[var(--dim)]">
              Agent 只能在工作区内直接操作文件；工作区外每次操作都需确认。
            </div>
          )}
        </div>
      )}
    </div>
  );
}