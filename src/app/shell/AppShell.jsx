import { useState } from 'react';
import TopBar from './TopBar';
import TaskList from '../../features/tasks/TaskList';
import WorkspacePanel from '../../features/workspace/WorkspacePanel';
import TaskTimeline from '../../features/tasks/TaskTimeline';
import TaskComposer from '../../features/tasks/TaskComposer';
import FileLogPanel from '../../features/activity/FileLogPanel';
import TrashPanel from '../../features/activity/TrashPanel';
import ApprovalQueue from '../../features/activity/ApprovalQueue';
import KnowledgeBasePanel from '../../features/knowledge-base/KnowledgeBasePanel';
import SessionPanel from '../../features/sessions/SessionPanel';
import ToolsPanel from '../../capabilities/tools/ToolsPanel';
import { useApp } from '../state/AppContext';

export default function AppShell() {
  const { state, dispatch } = useApp();
  const [modals, setModals] = useState({ kb: false, session: false, tools: false });

  const closeModal = (key) => setModals((m) => ({ ...m, [key]: false }));

  return (
    <div className="h-screen flex flex-col bg-[var(--bg)] text-[var(--text)]">
      <TopBar onOpenModal={(k) => setModals((m) => ({ ...m, [k]: true }))} />

      <div className="flex-1 flex overflow-hidden">
        {/* 左栏 */}
        <aside
          className="flex-shrink-0 border-r border-[var(--border)] bg-[var(--bg)] overflow-y-auto"
          style={{ width: 'var(--sidebar-w)' }}
        >
          <TaskList />
          <WorkspacePanel />
        </aside>

        {/* 中栏 */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
          <TaskTimeline />
          <TaskComposer />
        </main>

        {/* 右栏 */}
        {!state.rightCollapsed && (
          <aside
            className="flex-shrink-0 border-l border-[var(--border)] bg-[var(--bg)] overflow-y-auto"
            style={{ width: 'var(--right-w)' }}
          >
            <ApprovalQueue />
            <FileLogPanel />
            <TrashPanel />
          </aside>
        )}
      </div>

      <button
        onClick={() => dispatch({ type: 'TOGGLE_RIGHT' })}
        className="fixed right-3 bottom-3 z-30 w-8 h-8 bg-[var(--surface)] border border-[var(--border)] rounded-md text-[var(--muted)] hover:text-[var(--text)] hover:border-[var(--border-hover)] transition-colors"
        title={state.rightCollapsed ? '展开右栏' : '折叠右栏'}
      >
        {state.rightCollapsed ? '‹' : '›'}
      </button>

      <KnowledgeBasePanel isOpen={modals.kb} onClose={() => closeModal('kb')} />
      <SessionPanel
        isOpen={modals.session}
        onClose={() => closeModal('session')}
      />
      <ToolsPanel isOpen={modals.tools} onClose={() => closeModal('tools')} />
    </div>
  );
}