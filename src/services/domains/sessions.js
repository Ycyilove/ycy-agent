import { formData, requestJson } from '../http/client';

export const createSession = (name = '') => requestJson('/api/session/create', {
  method: 'POST',
  body: formData({ name }),
});
export const listSessions = () => requestJson('/api/session/list');
export const getSession = (sessionId) => requestJson(`/api/session/${sessionId}`);
export const deleteSession = (sessionId) => requestJson(`/api/session/${sessionId}`, { method: 'DELETE' });
export const renameSession = (sessionId, name) => requestJson(`/api/session/${sessionId}/rename`, {
  method: 'PUT',
  body: formData({ name }),
});
export const getSessionMessages = (sessionId) => requestJson(`/api/session/${sessionId}/messages`);
export const getSessionContext = (sessionId, maxMessages = 20) => requestJson(
  `/api/session/${sessionId}/context?max_messages=${maxMessages}`
);
export const addSessionMessage = (sessionId, role, content) => requestJson(
  `/api/session/${sessionId}/message`,
  { method: 'POST', body: formData({ role, content }) },
);
export const getSessionFiles = (sessionId) => requestJson(`/api/session/${sessionId}/files`);
export const addSessionFile = (sessionId, fileName, filePath = null, fileType = null) => requestJson(
  `/api/session/${sessionId}/file`,
  {
    method: 'POST',
    body: formData({ file_name: fileName, file_path: filePath, file_type: fileType }),
  },
);
export const deleteSessionFile = (fileId) => requestJson(`/api/session/file/${fileId}`, {
  method: 'DELETE',
});
export const searchSessionMessages = (sessionId, keyword, limit = 10) => {
  const params = new URLSearchParams({ session_id: sessionId, keyword, limit: String(limit) });
  return requestJson(`/api/session/search?${params}`);
};
