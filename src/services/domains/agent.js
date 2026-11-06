import { formData, openSseStream, requestJson } from '../http/client';

export const streamAgentTask = async ({
  taskId,
  message,
  history = [],
  mode = 'agent',
  workspace = [],
  sessionId = null,
  files = [],
  autoDiscoverMcp = false,
  signal,
}) => {
  const data = formData({
    task_id: taskId || '',
    message: message || '',
    history: JSON.stringify(history),
    mode,
    workspace: JSON.stringify(workspace),
    session_id: sessionId,
    auto_discover_mcp: String(autoDiscoverMcp),
  });
  for (const file of files) data.append('files', file);
  return openSseStream('/api/agent/tasks/stream', data, signal);
};

export const approveAgentTask = async (
  taskId,
  approvalId,
  action = 'allow',
  signal,
  sessionId,
) => {
  if (!sessionId) {
    // 前端调用方必须传 sessionId，否则后端 422（session_id 是必填 Form）
    throw new Error('approveAgentTask: sessionId 不能为空');
  }
  return openSseStream(
    `/api/agent/tasks/${taskId}/approve`,
    formData({ approval_id: approvalId, action, session_id: sessionId }),
    signal,
  );
};

export const listAgentTasks = (sessionId) => requestJson(
  `/api/agent/tasks?session_id=${encodeURIComponent(sessionId)}`
);
export const deleteAgentTask = (taskId, sessionId) => requestJson(
  `/api/agent/tasks/${taskId}?session_id=${encodeURIComponent(sessionId)}`,
  { method: 'DELETE' },
);