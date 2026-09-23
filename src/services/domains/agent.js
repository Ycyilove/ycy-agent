import { formData, openSseStream, requestJson } from '../http/client';

export const streamAgentTask = async ({
  taskId,
  message,
  history = [],
  mode = 'plan',
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

export const approveAgentTask = async (taskId, approvalId, action = 'allow', signal) => (
  openSseStream(
    `/api/agent/tasks/${taskId}/approve`,
    formData({ approval_id: approvalId, action }),
    signal,
  )
);

export const listAgentTasks = (sessionId) => requestJson(
  `/api/agent/tasks?session_id=${encodeURIComponent(sessionId)}`
);
export const getAgentTask = (taskId) => requestJson(`/api/agent/tasks/${taskId}`);
export const deleteAgentTask = (taskId) => requestJson(`/api/agent/tasks/${taskId}`, {
  method: 'DELETE',
});