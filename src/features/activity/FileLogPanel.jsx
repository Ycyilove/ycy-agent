import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { useApp } from '../../app/state/AppContext';

const ACTION_LABEL = {
  read: '读取',
  write: '写入',
  delete: '删除',
  restore: '恢复',
  move: '移动',
};

export default function FileLogPanel() {
  const { state } = useApp();

  return (
    <div className="border-b border-[var(--border)]">
      <div className="px-4 pt-4 pb-2">
        <span className="text-[11px] uppercase tracking-wider text-[var(--dim)]">文件操作</span>
      </div>
      <div className="px-3 pb-3 max-h-[280px] overflow-y-auto">
        {state.fileLog.length === 0 ? (
          <div className="px-1 py-2 text-xs text-[var(--dim)]">暂无操作记录</div>
        ) : (
          <div className="space-y-0.5">
            {state.fileLog.map(entry => (
              <div key={entry.id} className="px-1.5 py-1.5 rounded hover:bg-[var(--surface)] transition-colors">
                <div className="flex items-center gap-1.5">
                  <span className="font-mono text-[10px] text-[var(--dim)]">
                    {new Date(entry.time).toLocaleTimeString('zh-CN', { hour12: false })}
                  </span>
                  <span className="text-[10px] text-[var(--muted)]">{ACTION_LABEL[entry.action] || entry.action}</span>
                  {entry.status === 'success' && <Icon d={I.check} className="w-2.5 h-2.5 text-[var(--success)]" />}
                  {entry.status === 'failed' && <Icon d={I.close} className="w-2.5 h-2.5 text-[var(--danger)]" />}
                </div>
                <div className="font-mono text-[10px] text-[var(--muted)] truncate mt-0.5" title={entry.path}>
                  {entry.path}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
