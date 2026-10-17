import { requestJson } from '../http/client';

export const listTrash = () => requestJson('/api/trash/list');

export const restoreTrashItem = (id, sessionId) =>
  requestJson('/api/trash/restore', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, session_id: sessionId }),
  });

export const purgeTrashItem = (id, sessionId) =>
  requestJson('/api/trash/purge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, session_id: sessionId }),
  });

export const purgeAllTrash = () =>
  requestJson('/api/trash/purge_all', { method: 'POST' });