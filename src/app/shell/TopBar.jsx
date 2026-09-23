import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { useApp } from '../../app/state/AppContext';

const MODES = [
  {
    id: 'ask',
    label: 'Ask',
    hint: 'Chat only. No tools. Knowledge base lookup is allowed.',
  },
  {
    id: 'plan',
    label: 'Plan',
    hint: 'Agent plans and asks before executing.',
  },
  {
    id: 'craft',
    label: 'Auto',
    hint: 'Agent executes directly.',
  },
];

export default function TopBar({ onOpenModal }) {
  const { state, dispatch } = useApp();

  const session = state.activeSessionId
    ? state.sessions[state.activeSessionId]
    : null;
  const activeTaskId = session?.activeTaskId;
  const activeTask = activeTaskId ? state.tasks[activeTaskId] : null;

  return (
    <header className="flex-shrink-0 h-12 border-b border-[var(--border)] bg-[var(--bg)]">
      <div className="h-full px-4 flex items-center justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-[var(--accent)] rounded-md flex items-center justify-center">
              <Icon d={I.spark} className="w-3.5 h-3.5 text-[var(--bg)]" />
            </div>
            <span className="text-sm font-medium">Agent</span>
          </div>

          {activeTask && (
            <>
              <span className="text-[var(--dim)]">/</span>
              <span className="text-sm text-[var(--muted)] truncate max-w-[280px]">
                {activeTask.title}
              </span>
            </>
          )}
        </div>

        <div className="flex items-center bg-[var(--surface)] border border-[var(--border)] rounded-md p-0.5">
          {MODES.map((m) => (
            <button
              key={m.id}
              onClick={() => dispatch({ type: 'SET_MODE', mode: m.id })}
              title={m.hint}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                state.mode === m.id
                  ? 'bg-[var(--accent)] text-[var(--bg)] font-medium'
                  : 'text-[var(--muted)] hover:text-[var(--text)]'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={() => onOpenModal('session')}
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--surface)] rounded-md transition-colors"
          >
            <Icon d={I.chat} className="w-3.5 h-3.5" /> 会话
          </button>
          <button
            onClick={() => onOpenModal('kb')}
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--surface)] rounded-md transition-colors"
          >
            <Icon d={I.book} className="w-3.5 h-3.5" /> 知识库
          </button>
          <button
            onClick={() => onOpenModal('tools')}
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--surface)] rounded-md transition-colors"
          >
            <Icon d={I.tool} className="w-3.5 h-3.5" /> 工具
          </button>
          <div className="w-px h-4 bg-[var(--border)] mx-1" />
          <div className="flex items-center gap-1.5 px-1.5">
            <span className="w-1.5 h-1.5 bg-[var(--accent)] rounded-full" />
            <span className="text-xs text-[var(--muted)]">在线</span>
          </div>
        </div>
      </div>
    </header>
  );
}