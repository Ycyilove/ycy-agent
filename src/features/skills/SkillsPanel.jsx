import { useState, useEffect } from 'react';
import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import {
  listSkills, createSkill, updateSkill,
  deleteSkill, toggleSkill,
  getSessionSkills, putSessionSkills,
} from '../../services/api';

const EMPTY_FORM = {
  id: '', name: '', description: '', prompt_template: '',
  tool_names: '', parameters: '{}', examples: '',
  category: 'skill', danger_level: 'safe', enabled: true,
};

export default function SkillsPanel({ isOpen, onClose, sessionId }) {
  const [skills, setSkills] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState('');
  const [form, setForm] = useState(EMPTY_FORM);
  const [editingId, setEditingId] = useState(null);
  const [sessionEnabled, setSessionEnabled] = useState([]);

  useEffect(() => {
    if (isOpen) load();
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !sessionId) {
      setSessionEnabled([]);
      return;
    }
    getSessionSkills(sessionId)
      .then((d) => setSessionEnabled(d.enabled_skills || []))
      .catch(() => setSessionEnabled([]));
  }, [isOpen, sessionId]);

  const load = async () => {
    try {
      const data = await listSkills();
      setSkills(data.skills || []);
    } catch (e) {
      setStage('加载失败: ' + e.message);
    }
  };

  const resetForm = () => { setForm(EMPTY_FORM); setEditingId(null); };

  const submit = async () => {
    setLoading(true); setStage('');
    try {
      let paramsObj = {};
      try { paramsObj = JSON.parse(form.parameters || '{}'); }
      catch { throw new Error('parameters 不是合法 JSON'); }
      const payload = {
        ...form,
        tool_names: form.tool_names.split(',').map(s => s.trim()).filter(Boolean),
        examples: form.examples.split('\n').map(s => s.trim()).filter(Boolean),
        parameters: paramsObj,
      };
      if (editingId) await updateSkill(editingId, payload);
      else await createSkill(payload);
      setStage(editingId ? '已更新' : '已创建');
      resetForm();
      await load();
    } catch (e) {
      setStage('保存失败: ' + e.message);
    } finally { setLoading(false); }
  };

  const toggle = async (skill, enabled) => {
    if (!skill?.id) { setStage('技能缺少 id，请刷新列表'); return; }
    try { await toggleSkill(skill.id, enabled); await load(); }
    catch (e) { setStage('切换失败: ' + e.message); }
  };

  const remove = async (id) => {
    if (!id) { setStage('技能缺少 id'); return; }
    if (!confirm(`删除技能「${id}」？`)) return;
    try { await deleteSkill(id); await load(); }
    catch (e) { setStage('删除失败: ' + e.message); }
  };

  const edit = (skill) => {
    if (!skill?.id) { setStage('技能缺少 id'); return; }
    setEditingId(skill.id);
    setForm({
      id: skill.id,
      name: skill.name || '',
      description: skill.description || '',
      prompt_template: skill.prompt_template || '',
      tool_names: (skill.tool_names || []).join(', '),
      parameters: JSON.stringify(skill.parameters || {}, null, 2),
      examples: (skill.examples || []).join('\n'),
      category: skill.category || 'skill',
      danger_level: skill.danger_level || 'safe',
      enabled: skill.enabled !== false,
    });
  };

  const toggleForSession = async (skillId) => {
    if (!sessionId) return;
    const next = sessionEnabled.includes(skillId)
      ? sessionEnabled.filter((x) => x !== skillId)
      : [...sessionEnabled, skillId];
    setSessionEnabled(next);
    try { await putSessionSkills(sessionId, next); }
    catch (e) { setStage('会话技能更新失败: ' + e.message); }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="flex max-h-[80vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
        <div className="flex flex-shrink-0 items-center justify-between border-b border-[var(--border)] px-5 py-3">
          <h2 className="text-sm font-medium text-[var(--text)]">技能管理</h2>
          <button onClick={onClose} className="rounded p-1 text-[var(--muted)] hover:text-[var(--text)]">
            <Icon d={I.close} className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {stage && (
            <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)] p-2 text-xs text-[var(--muted)]">
              {stage}
            </div>
          )}

          {sessionId && (
            <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2 text-[11px] text-[var(--muted)]">
              当前会话：<span className="text-[var(--text)]">{sessionId.slice(0, 8)}…</span>
              {' '}— 勾选"本会话启用"的技能会在该会话任务中通过 prompt 引导生效。
            </div>
          )}

          {/* 表单 */}
          <div className="rounded-md border border-[var(--border)] p-3 space-y-2">
            <div className="text-xs text-[var(--muted)]">{editingId ? '编辑技能' : '新建技能'}</div>
            <input className="w-full h-8 rounded border border-[var(--border)] bg-[var(--surface-2)] px-2 text-xs"
              placeholder="id（唯一，创建后不可改）" value={form.id} disabled={!!editingId}
              onChange={(e) => setForm({ ...form, id: e.target.value })} />
            <input className="w-full h-8 rounded border border-[var(--border)] bg-[var(--surface-2)] px-2 text-xs"
              placeholder="名称" value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })} />
            <input className="w-full h-8 rounded border border-[var(--border)] bg-[var(--surface-2)] px-2 text-xs"
              placeholder="描述" value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })} />
            <textarea className="w-full rounded border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1 text-xs"
              rows={3} placeholder="prompt_template（注入到任务目标的引导文本）"
              value={form.prompt_template}
              onChange={(e) => setForm({ ...form, prompt_template: e.target.value })} />
            <input className="w-full h-8 rounded border border-[var(--border)] bg-[var(--surface-2)] px-2 text-xs"
              placeholder="tool_names（逗号分隔，仅作提示，不裁剪工具集）"
              value={form.tool_names}
              onChange={(e) => setForm({ ...form, tool_names: e.target.value })} />
            <textarea className="w-full rounded border border-[var(--border)] bg-[var(--surface-2)] px-2 py-1 text-xs font-mono"
              rows={3} placeholder='parameters JSON（如 {"query": {"type": "str"}}）'
              value={form.parameters}
              onChange={(e) => setForm({ ...form, parameters: e.target.value })} />
            <div className="flex items-center gap-2">
              <button disabled={loading} onClick={submit}
                className="rounded bg-[var(--accent)] px-3 py-1 text-xs text-[var(--bg)] disabled:opacity-50">
                {editingId ? '保存' : '创建'}
              </button>
              {editingId && (
                <button onClick={resetForm}
                  className="rounded border border-[var(--border)] px-3 py-1 text-xs text-[var(--muted)]">
                  取消
                </button>
              )}
            </div>
          </div>

          {/* 列表 */}
          {skills.map((s) => (
            <div key={s.id}
              className="flex items-start justify-between gap-3 rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="text-sm text-[var(--text)] truncate">
                  {s.name} <span className="text-[11px] text-[var(--dim)]">({s.id})</span>
                  {s.enabled === false && (
                    <span className="ml-1 text-[11px] text-[var(--danger)]">已禁用</span>
                  )}
                </div>
                <div className="text-[11px] text-[var(--muted)] truncate">{s.description}</div>
                <div className="text-[11px] text-[var(--dim)]">
                  工具: {(s.tool_names || []).join(', ') || '—'}
                </div>
              </div>
              <div className="flex flex-col items-end gap-1.5 flex-shrink-0">
                {sessionId && (
                  <label className="flex items-center gap-1 text-[11px] text-[var(--muted)] cursor-pointer">
                    <input type="checkbox"
                      checked={sessionEnabled.includes(s.id)}
                      onChange={() => toggleForSession(s.id)} />
                    本会话启用
                  </label>
                )}
                <div className="flex items-center gap-1.5">
                  <button onClick={() => toggle(s, !s.enabled)}
                    className="rounded border border-[var(--border)] px-2 py-1 text-[11px] text-[var(--muted)]">
                    {s.enabled ? '禁用' : '启用'}
                  </button>
                  <button onClick={() => edit(s)}
                    className="rounded border border-[var(--border)] px-2 py-1 text-[11px] text-[var(--muted)]">
                    编辑
                  </button>
                  <button onClick={() => remove(s.id)}
                    className="rounded p-1.5 text-[var(--dim)] hover:text-[var(--danger)]">
                    <Icon d={I.trash} className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
          {skills.length === 0 && (
            <div className="py-8 text-center text-xs text-[var(--dim)]">
              暂无技能，先创建一个
            </div>
          )}
        </div>

        <div className="flex flex-shrink-0 items-center justify-end border-t border-[var(--border)] px-5 py-3">
          <button onClick={onClose}
            className="rounded-md bg-[var(--accent)] px-3 py-1.5 text-xs font-medium text-[var(--bg)]">
            完成
          </button>
        </div>
      </div>
    </div>
  );
}