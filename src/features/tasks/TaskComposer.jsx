import { useCallback, useEffect, useRef, useState } from 'react';
import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { useApp } from '../../app/state/AppContext';
import { streamAgentTask } from '../../services/api';

const ACCEPT = 'image/*,.pdf,.txt,.docx,.xlsx,.csv,.json,.md';

export default function TaskComposer() {
  const { state, dispatch, registerStream, clearStream, abortStream } = useApp();
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isGlobalDragging, setIsGlobalDragging] = useState(false);
  const [submittingTasks, setSubmittingTasks] = useState(() => new Set());
  const [submitError, setSubmitError] = useState('');
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const dragCounter = useRef(0);

  const activeSessionId = state.activeSessionId;
  const session = activeSessionId ? state.sessions[activeSessionId] : null;
  const activeTaskId = session?.activeTaskId || null;
  const workspace = session?.workspace || { allowed: [], denied: [] };

  const isCurrentSubmitting = activeTaskId
    ? submittingTasks.has(activeTaskId)
    : false;

  const addFiles = useCallback((fileList) => {
    const incoming = Array.from(fileList || []);
    if (!incoming.length) return;
    setAttachments((previous) => {
      const seen = new Set(previous.map((file) => `${file.name}|${file.size}`));
      const next = [...previous];
      for (const file of incoming) {
        const key = `${file.name}|${file.size}`;
        if (!seen.has(key)) {
          seen.add(key);
          next.push(file);
        }
      }
      return next;
    });
  }, []);

  // 会话或任务切换时清空输入与错误
  useEffect(() => {
    setInput('');
    setAttachments([]);
    setSubmitError('');
  }, [activeSessionId, activeTaskId]);

  useEffect(() => {
    const handler = (event) => {
      setInput(event.detail);
      inputRef.current?.focus();
    };
    window.addEventListener('fill-composer', handler);
    return () => window.removeEventListener('fill-composer', handler);
  }, []);

  useEffect(() => {
    const onEnter = (event) => {
      event.preventDefault();
      if (!event.dataTransfer?.types?.includes('Files')) return;
      dragCounter.current += 1;
      setIsGlobalDragging(true);
    };
    const onOver = (event) => event.preventDefault();
    const onLeave = (event) => {
      event.preventDefault();
      dragCounter.current -= 1;
      if (dragCounter.current <= 0) {
        dragCounter.current = 0;
        setIsGlobalDragging(false);
      }
    };
    const onDrop = (event) => {
      event.preventDefault();
      dragCounter.current = 0;
      setIsGlobalDragging(false);
      addFiles(event.dataTransfer?.files);
    };

    window.addEventListener('dragenter', onEnter);
    window.addEventListener('dragover', onOver);
    window.addEventListener('dragleave', onLeave);
    window.addEventListener('drop', onDrop);
    return () => {
      window.removeEventListener('dragenter', onEnter);
      window.removeEventListener('dragover', onOver);
      window.removeEventListener('dragleave', onLeave);
      window.removeEventListener('drop', onDrop);
    };
  }, [addFiles]);

  const applyTaskEvent = useCallback(
    (taskId, event) => {
      if (event.type === 'task') {
        if (event.session_id) {
          dispatch({
            type: 'UPDATE_TASK',
            id: taskId,
            patch: { backendSessionId: event.session_id },
          });
        }
        return;
      }

      if (event.type === 'step' && event.step) {
        dispatch({ type: 'UPSERT_STEP', taskId, step: event.step });
        if (event.step.status === 'waiting' && event.step.approval) {
          dispatch({
            type: 'ADD_APPROVAL',
            item: { ...event.step.approval, taskId, stepId: event.step.id },
          });
        }
        return;
      }

      if (event.type === 'file_log' && event.entry) {
        dispatch({ type: 'ADD_FILE_LOG', entry: event.entry });
        return;
      }

      if (event.type === 'error') {
        dispatch({
          type: 'UPSERT_STEP',
          taskId,
          step: {
            id: `${taskId}:error`,
            label: event.message || '任务执行失败',
            status: 'failed',
            kind: 'think',
          },
        });
        dispatch({ type: 'UPDATE_TASK', id: taskId, patch: { status: 'failed' } });
        return;
      }

      if (event.type === 'done') {
        dispatch({
          type: 'UPDATE_TASK',
          id: taskId,
          patch: { status: event.task_status || 'done' },
        });
      }
    },
    [dispatch]
  );

  const submit = async (event) => {
    event.preventDefault();
    const message = input.trim();
    if ((!message && attachments.length === 0) || isCurrentSubmitting) return;

    setSubmitError('');

    if (!activeSessionId) {
      setSubmitError('请先创建或选择一个会话');
      return;
    }
    if (!session) {
      setSubmitError('会话不存在，请重新选择');
      return;
    }
    if (!activeTaskId) {
      setSubmitError('请先在左侧新建任务');
      return;
    }

    const taskId = activeTaskId;
    const files = [...attachments];
    const workspacePaths = workspace.allowed.map((dir) => dir.path);

    if (files.length > 0) {
      const currentTask = state.tasks[taskId];
      dispatch({
        type: 'UPDATE_TASK',
        id: taskId,
        patch: {
          attachments: [
            ...(currentTask?.attachments || []),
            ...files.map((file) => ({
              name: file.name,
              size: file.size,
              type: file.type,
            })),
          ],
        },
      });
    }

    dispatch({
      type: 'UPDATE_TASK',
      id: taskId,
      patch: { status: 'running' },
    });

    setInput('');
    setAttachments([]);
    setSubmittingTasks((prev) => {
      const next = new Set(prev);
      next.add(activeTaskId);
      return next;
    });

    const controller = new AbortController();
    registerStream(taskId, controller);

    const autoDiscoverMcp =
      localStorage.getItem('dsh.autoDiscoverMcpTools') === 'true';

    try {
      const stream = await streamAgentTask({
        taskId,
        message,
        mode: state.mode,
        workspace: workspacePaths,
        sessionId: activeSessionId,
        files,
        autoDiscoverMcp,
        signal: controller.signal,
      });
      for await (const taskEvent of stream.events()) {
        applyTaskEvent(taskId, taskEvent);
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        applyTaskEvent(taskId, {
          type: 'error',
          message: error.message || '任务执行失败',
        });
        setSubmitError(error.message || '任务执行失败');
      }
    } finally {
      clearStream(taskId);
      setSubmittingTasks((prev) => {
        const next = new Set(prev);
        next.delete(activeTaskId);
        return next;
      });
    }
  };

  const handleStop = () => {
    if (!activeTaskId) return;
    abortStream(activeTaskId);
    dispatch({
      type: 'UPDATE_TASK',
      id: activeTaskId,
      patch: { status: 'stopped' },
    });
    dispatch({
      type: 'UPSERT_STEP',
      taskId: activeTaskId,
      step: {
        id: `${activeTaskId}:stopped`,
        label: '用户已停止响应',
        status: 'failed',
        kind: 'think',
      },
    });
    setSubmittingTasks((prev) => {
      const next = new Set(prev);
      next.delete(activeTaskId);
      return next;
    });
  };

  const canSubmit =
    Boolean(activeTaskId) && (input.trim().length > 0 || attachments.length > 0);

  const placeholder = !activeSessionId
    ? '请先创建或选择一个会话…'
    : !activeTaskId
      ? '请先在左侧新建任务…'
      : isCurrentSubmitting
        ? '当前任务执行中，可继续补充说明…'
        : state.mode === 'ask'
          ? '问一个问题…'
          : '描述你想让 Agent 完成的任务…';

  return (
    <div className="relative flex-shrink-0">
      <div className="pointer-events-none absolute -top-8 left-0 right-0 h-8 bg-gradient-to-t from-[var(--bg)] to-transparent" />

      <div className="border-t border-[var(--border)] bg-[var(--bg)]">
        <div className="mx-auto max-w-[760px] px-6 py-4">
          {attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5">
              {attachments.map((file, index) => (
                <div
                  key={`${file.name}-${index}`}
                  className="group flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1 text-[11px] text-[var(--muted)]"
                >
                  <Icon d={I.doc} className="h-3 w-3 flex-shrink-0" />
                  <span className="max-w-[180px] truncate">{file.name}</span>
                  <button
                    type="button"
                    onClick={() =>
                      setAttachments((previous) =>
                        previous.filter((_, i) => i !== index)
                      )
                    }
                    className="text-[var(--dim)] transition-colors hover:text-[var(--danger)]"
                    title="移除附件"
                  >
                    <Icon d={I.close} className="h-2.5 w-2.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <form onSubmit={submit}>
            <div
              onDragOver={(event) => {
                event.preventDefault();
                event.stopPropagation();
                if (event.dataTransfer?.types?.includes('Files')) setIsDragging(true);
              }}
              onDragLeave={(event) => {
                event.preventDefault();
                event.stopPropagation();
                setIsDragging(false);
              }}
              onDrop={(event) => {
                event.preventDefault();
                event.stopPropagation();
                setIsDragging(false);
                addFiles(event.dataTransfer?.files);
              }}
              className={`relative flex items-center gap-2 rounded-lg transition-colors ${
                isDragging ? 'ring-1 ring-[var(--accent)]' : ''
              }`}
            >
              <div className="relative flex-1">
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder={placeholder}
                  disabled={!activeTaskId}
                  className={`h-10 w-full rounded-lg border bg-[var(--surface)] pl-3 pr-10 text-sm text-[var(--text)] placeholder-[var(--dim)] transition-colors focus:outline-none disabled:cursor-not-allowed disabled:opacity-60 ${
                    isDragging
                      ? 'border-[var(--accent)]'
                      : 'border-[var(--border)] focus:border-[var(--accent)]/60'
                  }`}
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={!activeTaskId}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1.5 text-[var(--dim)] transition-colors hover:text-[var(--muted)] disabled:opacity-40"
                  title="添加附件"
                >
                  <Icon d={I.clip} className="h-4 w-4" />
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept={ACCEPT}
                  onChange={(event) => {
                    addFiles(event.target.files);
                    event.target.value = '';
                  }}
                  className="hidden"
                />
              </div>

              {isCurrentSubmitting ? (
                <button
                  type="button"
                  onClick={handleStop}
                  className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-md border border-[var(--danger)]/40 bg-[var(--danger)]/10 text-[var(--danger)] transition-colors hover:bg-[var(--danger)]/20"
                  title="停止响应"
                >
                  <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="6" width="12" height="12" rx="1" />
                  </svg>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!canSubmit}
                  className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-md bg-[var(--accent)] text-[var(--bg)] transition-colors hover:bg-[var(--accent-hover)] disabled:cursor-not-allowed disabled:bg-[var(--border)] disabled:text-[var(--dim)]"
                  title="发送任务"
                >
                  <Icon d={I.send} className="h-4 w-4" />
                </button>
              )}
            </div>
          </form>

          {submitError && (
            <div className="mt-2 rounded-md border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-2.5 py-1.5 text-[11px] text-[var(--danger)]">
              {submitError}
            </div>
          )}

          {!submitError && !activeSessionId && (
            <div className="mt-2 text-[11px] text-[var(--dim)]">
              请先创建或选择一个会话
            </div>
          )}

          {!submitError && activeSessionId && !activeTaskId && (
            <div className="mt-2 text-[11px] text-[var(--dim)]">
              请先在左侧新建任务
            </div>
          )}

          {!submitError && isCurrentSubmitting && (
            <div className="mt-2 text-[11px] text-[var(--dim)]">
              当前任务执行中，可继续输入补充说明
            </div>
          )}
        </div>
      </div>

      {isGlobalDragging && (
        <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-black/70">
          <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-8 shadow-xl">
            <div className="flex flex-col items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-md border border-[var(--border)] bg-[var(--surface-2)]">
                <Icon d={I.folder} className="h-6 w-6 text-[var(--accent)]" />
              </div>
              <div className="text-sm text-[var(--text)]">释放以添加文件</div>
              <div className="text-xs text-[var(--dim)]">文件将作为附件添加到输入区</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}