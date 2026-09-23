import { useRef, useEffect } from 'react';
import TimelineStep from './TimelineStep';
import { useApp } from '../../app/state/AppContext';

export default function TaskTimeline() {
  const { state } = useApp();
  const session = state.activeSessionId
    ? state.sessions[state.activeSessionId]
    : null;
  const activeTaskId = session?.activeTaskId;
  const task = activeTaskId ? state.tasks[activeTaskId] : null;
  const workspace = session?.workspace || { allowed: [], denied: [] };
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [task?.steps?.length]);

  if (!task) {
    return (
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[760px] px-6 pt-[140px]">
          <h2 className="mb-2 text-xl font-semibold text-[var(--text)]">
            开始一个任务
          </h2>
          <p className="mb-8 max-w-md text-sm leading-relaxed text-[var(--muted)]">
            描述你想让 Agent 完成的本地文件操作。它会拆解步骤、逐步执行，关键操作会先征求你的确认。
          </p>

          <div className="space-y-4">
            <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <div className="mb-2 text-xs text-[var(--muted)]">工作区状态</div>
              <div className="text-sm text-[var(--text)]">
                {!session
                  ? '未选择会话'
                  : workspace.allowed.length === 0
                    ? '未授权任何目录'
                    : `已授权 ${workspace.allowed.length} 个目录`}
              </div>
            </div>

            <div>
              <div className="mb-2 text-xs text-[var(--muted)]">试试这样说</div>
              <div className="space-y-1.5">
                {[
                  '把桌面上的 Q2 数据.xlsx 合并成一张表',
                  '整理下载文件夹，按类型分类',
                  '读取 report.docx，生成摘要 PDF',
                ].map((text) => (
                  <button
                    key={text}
                    onClick={() =>
                      window.dispatchEvent(
                        new CustomEvent('fill-composer', { detail: text })
                      )
                    }
                    className="block w-full rounded-md px-3 py-2 text-left text-sm text-[var(--muted)] transition-colors hover:bg-[var(--surface)] hover:text-[var(--text)]"
                  >
                    · {text}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-[760px] px-6 py-6">
        <div className="mb-6">
          <div className="mb-1 flex items-center gap-2">
            <span className="text-xs text-[var(--dim)]">
              {session?.name || '任务'}
            </span>
          </div>
          <h2 className="text-lg font-medium text-[var(--text)]">{task.title}</h2>
        </div>

        <div className="space-y-0">
          {(task.steps || []).map((step, i) => (
            <TimelineStep
              key={step.id}
              step={step}
              taskId={task.id}
              isLast={i === task.steps.length - 1}
            />
          ))}
        </div>

        <div ref={endRef} />
      </div>
    </div>
  );
}