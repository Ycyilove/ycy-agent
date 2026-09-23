import { useApp } from '../../app/state/AppContext';

export default function ApprovalQueue() {
  const { state } = useApp();

  if (state.approvals.length === 0) return null;

  return (
    <div className="border-b border-[var(--border)]">
      <div className="px-4 pt-4 pb-2 flex items-center gap-2">
        <span className="text-[11px] uppercase tracking-wider text-[var(--dim)]">待确认</span>
        <span className="px-1.5 py-0.5 bg-[var(--warning)]/15 text-[var(--warning)] text-[10px] rounded">
          {state.approvals.length}
        </span>
      </div>
      <div className="px-3 pb-3 space-y-2">
        {state.approvals.map(a => (
          <div key={a.id} className="p-2.5 bg-[var(--surface-2)] border border-[var(--border)] rounded-md">
            <div className="text-xs text-[var(--text)]">{a.title}</div>
            {a.path && (
              <div className="mt-1 font-mono text-[10px] text-[var(--muted)] truncate" title={a.path}>
                {a.path}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
