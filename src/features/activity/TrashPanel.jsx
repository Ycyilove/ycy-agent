import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { useApp } from '../../app/state/AppContext';

export default function TrashPanel() {
  const { state, dispatch } = useApp();

  return (
    <div>
      <div className="px-4 pt-4 pb-2 flex items-center gap-2">
        <span className="text-[11px] uppercase tracking-wider text-[var(--dim)]">回收站</span>
        {state.trash.length > 0 && (
          <span className="px-1.5 py-0.5 bg-[var(--surface-2)] text-[var(--muted)] text-[10px] rounded">
            {state.trash.length}
          </span>
        )}
      </div>
      <div className="px-3 pb-4">
        {state.trash.length === 0 ? (
          <div className="px-1 py-2 text-xs text-[var(--dim)]">
            Agent 删除的文件会暂存在这里，7 天后自动清理。
          </div>
        ) : (
          <div className="space-y-1.5">
            {state.trash.map(item => (
              <div key={item.id} className="p-2 bg-[var(--surface-2)] border border-[var(--border)] rounded-md">
                <div className="text-xs text-[var(--text)] truncate">{item.name}</div>
                <div className="font-mono text-[10px] text-[var(--dim)] truncate mt-0.5" title={item.path}>
                  {item.path}
                </div>
                <div className="flex items-center gap-2 mt-1.5">
                  <button
                    onClick={() => dispatch({ type: 'RESTORE_TRASH', id: item.id })}
                    className="flex items-center gap-1 text-[10px] text-[var(--muted)] hover:text-[var(--success)] transition-colors"
                  >
                    <Icon d={I.restore} className="w-3 h-3" /> 恢复
                  </button>
                  <button
                    onClick={() => {
                      if (window.confirm(`永久删除「${item.name}」？此操作不可恢复。`)) {
                        dispatch({ type: 'PURGE_TRASH', id: item.id });
                      }
                    }}
                    className="text-[10px] text-[var(--muted)] hover:text-[var(--danger)] transition-colors"
                  >
                    永久删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
