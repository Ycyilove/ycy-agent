import { useEffect, useState } from 'react';
import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { useApp } from '../../app/state/AppContext';
import {
  listTrash,
  restoreTrashItem,
  purgeTrashItem,
} from '../../services/api';

export default function TrashPanel() {
  const { state, dispatch } = useApp();
  const activeSessionId = state.activeSessionId;
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState(null);

  const refresh = async () => {
    setLoading(true);
    try {
      const data = await listTrash();
      setItems(data.items || []);
    } catch (err) {
      console.warn('[dsh][trash] list failed', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const handler = () => refresh();
    window.addEventListener('trash-updated', handler);
    return () => window.removeEventListener('trash-updated', handler);
  }, []);

  const handleRestore = async (item) => {
    if (busyId) return;
    setBusyId(item.id);
    try {
      const result = await restoreTrashItem(item.id, activeSessionId);
      if (result.path_adjusted) {
        window.alert(result.message);
      }
      dispatch({
        type: 'ADD_FILE_LOG',
        entry: {
          id: `log-${Date.now()}`,
          action: 'restore',
          path: result.restored_path || item.original_path,
          time: Date.now(),
          status: 'success',
        },
      });
      await refresh();
    } catch (err) {
      window.alert(err.message || '恢复失败');
    } finally {
      setBusyId(null);
    }
  };

  const handlePurge = async (item) => {
    if (busyId) return;
    if (!window.confirm(`永久删除「${item.original_name}」？此操作不可恢复。`)) return;
    setBusyId(item.id);
    try {
      await purgeTrashItem(item.id, activeSessionId);
      dispatch({
        type: 'ADD_FILE_LOG',
        entry: {
          id: `log-${Date.now()}`,
          action: 'delete',
          path: item.original_path,
          time: Date.now(),
          status: 'success',
        },
      });
      await refresh();
    } catch (err) {
      window.alert(err.message || '删除失败');
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div>
      <div className="px-4 pt-4 pb-2 flex items-center gap-2">
        <span className="text-[11px] uppercase tracking-wider text-[var(--dim)]">
          回收站
        </span>
        {items.length > 0 && (
          <span className="px-1.5 py-0.5 bg-[var(--surface-2)] text-[var(--muted)] text-[10px] rounded">
            {items.length}
          </span>
        )}
        <button
          onClick={refresh}
          disabled={loading}
          className="ml-auto rounded p-0.5 text-[var(--dim)] hover:text-[var(--text)] disabled:opacity-40"
          title="刷新"
        >
          <Icon
            d={I.restore}
            className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`}
          />
        </button>
      </div>
      <div className="px-3 pb-4">
        {items.length === 0 ? (
          <div className="px-1 py-2 text-xs text-[var(--dim)]">
            Agent 删除的文件会暂存在这里，可恢复或永久删除。
          </div>
        ) : (
          <div className="space-y-1.5">
            {items.map((item) => (
              <div
                key={item.id}
                className="p-2 bg-[var(--surface-2)] border border-[var(--border)] rounded-md"
              >
                <div className="text-xs text-[var(--text)] truncate">
                  {item.original_name}
                </div>
                <div
                  className="font-mono text-[10px] text-[var(--dim)] truncate mt-0.5"
                  title={item.original_path}
                >
                  {item.original_path}
                </div>
                <div className="flex items-center gap-2 mt-1.5">
                  <button
                    onClick={() => handleRestore(item)}
                    disabled={busyId === item.id}
                    className="flex items-center gap-1 text-[10px] text-[var(--muted)] hover:text-[var(--success)] transition-colors disabled:opacity-40"
                  >
                    <Icon d={I.restore} className="w-3 h-3" /> 恢复
                  </button>
                  <button
                    onClick={() => handlePurge(item)}
                    disabled={busyId === item.id}
                    className="text-[10px] text-[var(--muted)] hover:text-[var(--danger)] transition-colors disabled:opacity-40"
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