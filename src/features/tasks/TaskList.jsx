import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { useApp } from '../../app/state/AppContext';
import { deleteAgentTask } from '../../services/api';

const STATUS = {
  idle: { dot: 'bg-[var(--dim)]' },
  running: { dot: 'bg-[var(--accent)] animate-pulse' },
  waiting: { dot: 'bg-[var(--warning)]' },
  interrupted: { dot: 'bg-[var(--warning)]' },
  cancelled: { dot: 'bg-[var(--dim)]' },
  done: { dot: 'bg-[var(--success)]' },
  failed: { dot: 'bg-[var(--danger)]' },
};

export default function TaskList() {
  const { state, dispatch } = useApp();

  const session = state.activeSessionId
    ? state.sessions[state.activeSessionId]
    : null;
  const taskIds = session?.taskIds || [];
  const tasks = taskIds.map((id) => state.tasks[id]).filter(Boolean);

  const running = tasks.filter(
    (t) => t.status === 'running' || t.status === 'waiting'
  );
  const finished = tasks.filter(
    (t) =>
      t.status === 'done' ||
      t.status === 'failed' ||
      t.status === 'idle' ||
      t.status === 'interrupted' ||
      t.status === 'cancelled'
  );

  const createTask = () => {
    if (!state.activeSessionId) return;
    const session = state.sessions[state.activeSessionId];
    if (!session) return;
    const id = `task-${Date.now()}`;
    const workspace = session.workspace || { allowed: [] };
    dispatch({
      type: 'ADD_TASK',
      task: {
        id,
        sessionId: state.activeSessionId,
        title: '新任务',
        status: 'idle',       // ← 尚未提交
        steps: [],
        workspace: workspace.allowed.map((d) => d.path),
        createdAt: Date.now(),
      },
    });
  };

  return (
    <div className="border-b border-[var(--border)]">
      <div className="flex items-center justify-between px-4 pb-2 pt-4">
        <span className="text-[11px] uppercase tracking-wider text-[var(--dim)]">
          任务
        </span>
        <button
          onClick={createTask}
          disabled={!state.activeSessionId}
          className="rounded p-1 text-[var(--muted)] transition-colors hover:text-[var(--text)] disabled:opacity-40"
          title="新建任务"
        >
          <Icon d={I.plus} className="h-3.5 w-3.5" />
        </button>
      </div>

      {running.length > 0 && (
        <div className="space-y-0.5 px-2 pb-2">
          {running.map((t) => (
            <TaskItem
              key={t.id}
              task={t}
              isActive={t.id === session?.activeTaskId}
            />
          ))}
        </div>
      )}

      {finished.length > 0 && (
        <>
          <div className="px-4 py-2 text-[11px] uppercase tracking-wider text-[var(--dim)]">
            已完成
          </div>
          <div className="space-y-0.5 px-2 pb-3">
            {finished.map((t) => (
              <TaskItem
                key={t.id}
                task={t}
                isActive={t.id === session?.activeTaskId}
              />
            ))}
          </div>
        </>
      )}

      {tasks.length === 0 && (
        <div className="px-4 pb-4 text-xs text-[var(--dim)]">
          {state.activeSessionId
            ? '暂无任务，点击 + 新建'
            : '请先创建或选择一个会话'}
        </div>
      )}
    </div>
  );
}

function TaskItem({ task, isActive }) {
  const { dispatch } = useApp();
  const s = STATUS[task.status] || STATUS.running;

  const handleDelete = async (event) => {
    // 阻止冒泡，避免触发外层“选中任务”
    event.stopPropagation();
    if (!window.confirm(`删除任务「${task.title}」？`)) return;

    // 本地任务未提交到后端（id 以 task- 开头且无 createdAt 服务端字段）
    // 直接本地删除即可
    try {
      await deleteAgentTask(task.id, task.sessionId);
    } catch (error) {
      // 404 说明后端已不存在，本地删除仍然正确
      const status = error?.status ?? error?.response?.status;
      if (status !== 404) {
        console.error('[dsh][task] 删除任务失败', error);
        return;
      }
    }

    dispatch({ type: 'DELETE_TASK', id: task.id });
  };

  return (
    <div
      className={`group flex w-full items-center gap-1 rounded-md border px-2.5 py-2 transition-colors ${
        isActive
          ? 'border-[var(--border)] bg-[var(--surface-2)]'
          : 'border-transparent hover:bg-[var(--surface)]'
      }`}
    >
      <button
        onClick={() =>
          dispatch({
            type: 'SET_ACTIVE_TASK',
            sessionId: task.sessionId,
            taskId: task.id,
          })
        }
        className="min-w-0 flex-1 text-left"
      >
        <div className="flex min-w-0 items-center gap-2">
          <span className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${s.dot}`} />
          <span
            className={`truncate text-xs ${
              isActive ? 'text-[var(--text)]' : 'text-[var(--muted)]'
            }`}
          >
            {task.title}
          </span>
        </div>
        {task.steps?.length > 0 && (
          <div className="mt-0.5 truncate pl-3.5 text-[11px] text-[var(--dim)]">
            {task.steps[task.steps.length - 1].label}
          </div>
        )}
      </button>

      <button
        onClick={handleDelete}
        className="flex-shrink-0 rounded p-1 text-[var(--dim)] opacity-0 transition-opacity hover:text-[var(--danger)] group-hover:opacity-100"
        title="删除任务"
        aria-label="删除任务"
      >
        <Icon d={I.trash} className="h-3 w-3" />
      </button>
    </div>
  );
}