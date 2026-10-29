import { requestJson } from '../http/client';

const BASE = '/api/skills';

export async function listSkills() {
  return requestJson(BASE, { method: 'GET' });
}
export async function getSkill(id) {
  return requestJson(`${BASE}/${encodeURIComponent(id)}`, { method: 'GET' });
}
export async function createSkill(payload) {
  return requestJson(BASE, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}
export async function updateSkill(id, payload) {
  return requestJson(`${BASE}/${encodeURIComponent(id)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}
export async function deleteSkill(id) {
  return requestJson(`${BASE}/${encodeURIComponent(id)}`, { method: 'DELETE' });
}
export async function toggleSkill(id, enabled) {
  return requestJson(`${BASE}/${encodeURIComponent(id)}/toggle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
}
export async function reloadSkills() {
  return requestJson(`${BASE}/reload`, { method: 'POST' });
}
export async function getSessionSkills(sessionId) {
  return requestJson(`/api/sessions/${encodeURIComponent(sessionId)}/skills`);
}
export async function putSessionSkills(sessionId, ids) {
  return requestJson(`/api/sessions/${encodeURIComponent(sessionId)}/skills`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled_skills: ids }),
  });
}