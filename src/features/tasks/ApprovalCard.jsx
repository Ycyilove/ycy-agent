import { useState } from 'react';
import { approveAgentTask } from '../../services/api';
import { useApp } from '../../app/state/AppContext';

export default function ApprovalCard({ approval, taskId, stepId }) {
  const { dispatch } = useApp();
  const [resolved, setResolved] = useState(false);
  const [busy, setBusy] = useState(false);

  const resolve = async (requestedAction) => {
    if (busy) return;
    const resolvedAction = requestedAction === 'save-as' ? 'allow' : requestedAction;
    const currentTaskId = approval.taskId || taskId;
    const currentStepId = approval.stepId || stepId;
    setBusy(true);
    setResolved(true);
    dispatch({ type: 'RESOLVE_APPROVAL', id: approval.id });

    try {
      const stream = await approveAgentTask(currentTaskId, approval.id, resolvedAction);
      for await (const event of stream.events()) {
        if (event.type === 'step' && event.step) {
          dispatch({ type: 'UPSERT_STEP', taskId: currentTaskId, step: event.step });
        } else if (event.type === 'file_log' && event.entry) {
          dispatch({ type: 'ADD_FILE_LOG', entry: event.entry });
        } else if (event.type === 'done') {
          dispatch({
            type: 'UPDATE_TASK',
            id: currentTaskId,
            patch: { status: event.task_status || 'done' },
          });
        } else if (event.type === 'error') {
          dispatch({
            type: 'UPDATE_STEP',
            taskId: currentTaskId,
            stepId: currentStepId,
            patch: { status: 'failed', label: event.message || '审批执行失败' },
          });
          dispatch({ type: 'UPDATE_TASK', id: currentTaskId, patch: { status: 'failed' } });
        }
      }
    } catch (error) {
      dispatch({
        type: 'UPDATE_STEP',
        taskId: currentTaskId,
        stepId: currentStepId,
        patch: { status: 'failed', label: error.message || '审批执行失败' },
      });
      dispatch({ type: 'UPDATE_TASK', id: currentTaskId, patch: { status: 'failed' } });
    } finally {
      setBusy(false);
    }
  };

  if (resolved) return null;

  return (
    <div className="rounded-md border border-[var(--warning)]/40 bg-[var(--warning)]/5 p-3">
      <div className="mb-2 flex items-start gap-2">
        <span className="mt-0.5 text-[var(--warning)]">⚠</span>
        <div className="min-w-0 flex-1">
          <div className="text-sm text-[var(--text)]">{approval.title}</div>
          {approval.path && (
            <div className="mt-1 truncate font-mono text-[11px] text-[var(--muted)]" title={approval.path}>
              {approval.path}
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={() => resolve('allow')}
          disabled={busy}
          className="rounded-md bg-[var(--warning)] px-3 py-1 text-xs font-medium text-[var(--bg)] transition-colors hover:bg-[var(--accent-hover)] disabled:opacity-50"
        >
          允许
        </button>
        <button
          onClick={() => resolve('deny')}
          disabled={busy}
          className="rounded-md border border-[var(--border)] px-3 py-1 text-xs text-[var(--muted)] transition-colors hover:border-[var(--border-hover)] hover:text-[var(--text)] disabled:opacity-50"
        >
          拒绝
        </button>
        {approval.canSaveAs && (
          <button
            onClick={() => resolve('save-as')}
            disabled={busy}
            className="rounded-md border border-[var(--border)] px-3 py-1 text-xs text-[var(--muted)] transition-colors hover:border-[var(--border-hover)] hover:text-[var(--text)] disabled:opacity-50"
          >
            另存为…
          </button>
        )}
        <button
          onClick={() => resolve('allow-always')}
          disabled={busy}
          className="ml-auto text-[11px] text-[var(--dim)] transition-colors hover:text-[var(--muted)] disabled:opacity-50"
        >
          本次会话始终允许
        </button>
      </div>
    </div>
  );
}
