import { useReducer, useRef, useCallback, useEffect, useContext } from 'react';
import { AppContext } from './AppContext';
import { listAgentTasks } from '../../services/api';

const EMPTY_WORKSPACE = { allowed: [], denied: [] };

const initialState = {
  sessions: {},
  tasks: {},
  activeSessionId: null,
  modelName: '',
  mode: 'ask',
  fileLog: [],
  trash: [],
  approvals: [],
  rightCollapsed: false,
  // 已被前端处理过的 approval id 集合。
  // 客户端权威，不会被后端 SSE 覆盖。
  resolvedApprovalIds: new Set(),
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_MODE':
      return { ...state, mode: action.mode };

    case 'SET_MODEL':
      return { ...state, modelName: action.modelName };

    case 'CREATE_SESSION': {
      const { id, name } = action;
      const existing = state.sessions[id];
      if (existing) {
        return {
          ...state,
          activeSessionId: state.activeSessionId ?? id,
        };
      }
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [id]: {
            id,
            name: name || '新会话',
            taskIds: [],
            activeTaskId: null,
            workspace: { ...EMPTY_WORKSPACE },   // ← 会话级工作区，默认空
          },
        },
        activeSessionId: state.activeSessionId ?? id,
      };
    }

    case 'DELETE_SESSION': {
      const session = state.sessions[action.id];
      if (!session) return state;
      const tasks = { ...state.tasks };
      for (const taskId of session.taskIds) delete tasks[taskId];
      const sessions = { ...state.sessions };
      delete sessions[action.id];
      const remainingIds = Object.keys(sessions);
      return {
        ...state,
        sessions,
        tasks,
        activeSessionId:
          state.activeSessionId === action.id
            ? remainingIds[0] || null
            : state.activeSessionId,
      };
    }

    case 'SET_ACTIVE_SESSION':
      return { ...state, activeSessionId: action.id };

    case 'SET_SESSION_NAME': {
      const session = state.sessions[action.id];
      if (!session) return state;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [action.id]: { ...session, name: action.name },
        },
      };
    }

    case 'ADD_TASK': {
      const { task } = action;
      const session = state.sessions[task.sessionId];
      if (!session) return state;
      return {
        ...state,
        tasks: { ...state.tasks, [task.id]: task },
        sessions: {
          ...state.sessions,
          [task.sessionId]: {
            ...session,
            taskIds: session.taskIds.includes(task.id)
              ? session.taskIds
              : [...session.taskIds, task.id],
            activeTaskId: task.id,
          },
        },
      };
    }

    case 'UPDATE_TASK': {
      const task = state.tasks[action.id];
      if (!task) return state;
      return {
        ...state,
        tasks: { ...state.tasks, [action.id]: { ...task, ...action.patch } },
      };
    }

    case 'SET_ACTIVE_TASK': {
      const session = state.sessions[action.sessionId];
      if (!session) return state;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [action.sessionId]: { ...session, activeTaskId: action.taskId },
        },
      };
    }

    case 'APPEND_STEP': {
      const { taskId, step } = action;
      const task = state.tasks[taskId];
      if (!task) return state;
      return {
        ...state,
        tasks: {
          ...state.tasks,
          [taskId]: { ...task, steps: [...(task.steps || []), step] },
        },
      };
    }

    case 'UPSERT_STEP': {
      const { taskId, step } = action;
      const task = state.tasks[taskId];
      if (!task) return state;
      const steps = task.steps || [];
      const exists = steps.some((item) => item.id === step.id);
      return {
        ...state,
        tasks: {
          ...state.tasks,
          [taskId]: {
            ...task,
            steps: exists
              ? steps.map((item) =>
                  item.id === step.id ? { ...item, ...step } : item
                )
              : [...steps, step],
          },
        },
      };
    }

    case 'UPDATE_STEP': {
      const { taskId, stepId, patch } = action;
      const task = state.tasks[taskId];
      if (!task) return state;
      return {
        ...state,
        tasks: {
          ...state.tasks,
          [taskId]: {
            ...task,
            steps: (task.steps || []).map((step) =>
              step.id === stepId ? { ...step, ...patch } : step
            ),
          },
        },
      };
    }

    case 'HYDRATE_TASKS': {
      const { sessionId, tasks } = action;
      const session = state.sessions[sessionId];
      if (!session) return state;

      const newTasks = { ...state.tasks };
      const taskIds = [];
      for (const task of tasks) {
        newTasks[task.id] = task;
        taskIds.push(task.id);
      }

      return {
        ...state,
        tasks: newTasks,
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            taskIds,
            activeTaskId:
              session.activeTaskId ||
              taskIds[taskIds.length - 1] ||
              null,
          },
        },
      };
    }

    case 'ADD_FILE_LOG':
      return {
        ...state,
        fileLog: [action.entry, ...state.fileLog].slice(0, 200),
      };

    case 'ADD_TRASH':
      return { ...state, trash: [...state.trash, action.item] };

    case 'RESTORE_TRASH': {
      const item = state.trash.find((t) => t.id === action.id);
      if (!item) return state;
      return {
        ...state,
        trash: state.trash.filter((t) => t.id !== action.id),
        fileLog: [
          {
            id: `log-${Date.now()}`,
            action: 'restore',
            path: item.path,
            time: Date.now(),
            status: 'success',
          },
          ...state.fileLog,
        ],
      };
    }

    case 'PURGE_TRASH':
      return {
        ...state,
        trash: state.trash.filter((item) => item.id !== action.id),
      };

    case 'ADD_APPROVAL':
      return {
        ...state,
        approvals: [
          ...state.approvals.filter((item) => item.id !== action.item.id),
          action.item,
        ],
      };

    case 'RESOLVE_APPROVAL': {
      const nextResolved = new Set(state.resolvedApprovalIds);
      nextResolved.add(action.id);
      return {
        ...state,
        approvals: state.approvals.filter((item) => item.id !== action.id),
        resolvedApprovalIds: nextResolved,
      };
    }

    // ── 工作区：作用在会话上 ──
    case 'ADD_WORKSPACE_DIR': {
      const sessionId = action.sessionId ?? state.activeSessionId;
      const session = state.sessions[sessionId];
      if (!session) return state;
      const workspace = session.workspace || EMPTY_WORKSPACE;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            workspace: {
              ...workspace,
              allowed: [...workspace.allowed, action.dir],
            },
          },
        },
      };
    }

    case 'REMOVE_WORKSPACE_DIR': {
      const sessionId = action.sessionId ?? state.activeSessionId;
      const session = state.sessions[sessionId];
      if (!session) return state;
      const workspace = session.workspace || EMPTY_WORKSPACE;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...session,
            workspace: {
              ...workspace,
              allowed: workspace.allowed.filter(
                (dir) => dir.path !== action.path
              ),
            },
          },
        },
      };
    }

    case 'TOGGLE_RIGHT':
      return { ...state, rightCollapsed: !state.rightCollapsed };

    default:
      return state;
  }
}

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const streamsRef = useRef(new Map());
  const hydratedSessionsRef = useRef(new Set());

  const registerStream = useCallback((taskId, controller) => {
    const existing = streamsRef.current.get(taskId);
    if (existing) existing.abort();
    streamsRef.current.set(taskId, controller);
  }, []);

  const clearStream = useCallback((taskId) => {
    streamsRef.current.delete(taskId);
  }, []);

  const abortStream = useCallback((taskId) => {
    const controller = streamsRef.current.get(taskId);
    if (controller) {
      controller.abort();
      streamsRef.current.delete(taskId);
    }
  }, []);

  // activeSessionId 变化时拉取该会话的任务
  useEffect(() => {
    const sessionId = state.activeSessionId;
    if (!sessionId) return;
    if (hydratedSessionsRef.current.has(sessionId)) return;
    if (!state.sessions[sessionId]) return;

    hydratedSessionsRef.current.add(sessionId);

    listAgentTasks(sessionId)
      .then((data) => {
        const tasks = (data.tasks || []).map((task) => ({
          ...task,
          id: task.id || task.task_id,
          sessionId: task.sessionId || task.session_id,
          steps: task.steps || [],
          createdAt: task.createdAt || task.created_at,
        }));
        console.log('[dsh][hydrate] 恢复任务', {
          sessionId,
          taskCount: tasks.length,
        });
        dispatch({ type: 'HYDRATE_TASKS', sessionId, tasks });
      })
      .catch((error) => {
        console.warn('[dsh][hydrate] 拉取任务失败', { sessionId, error });
        hydratedSessionsRef.current.delete(sessionId);
      });
  }, [state.activeSessionId, state.sessions]);

  return (
    <AppContext.Provider
      value={{ state, dispatch, registerStream, clearStream, abortStream }}
    >
      {children}
    </AppContext.Provider>
  );
}

// ── 会话级工作区读取 hook ──
export function useActiveWorkspace() {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error('useActiveWorkspace must be used within AppProvider');
  }
  const session = ctx.state.activeSessionId
    ? ctx.state.sessions[ctx.state.activeSessionId]
    : null;
  return session?.workspace || EMPTY_WORKSPACE;
}