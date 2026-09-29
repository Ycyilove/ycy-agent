import { useEffect, useReducer, useCallback } from 'react';
import { MCPContext } from './MCPContext';

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const initialState = {
  servers: [],
  loading: false,
  error: null,
  lastFetchedAt: null,
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_SERVERS':
      return {
        ...state,
        servers: action.servers,
        loading: false,
        error: null,
        lastFetchedAt: Date.now(),
      };
    case 'SET_LOADING':
      return { ...state, loading: action.loading };
    case 'SET_ERROR':
      return { ...state, error: action.error, loading: false };
    default:
      return state;
  }
}

async function fetchMcpServers() {
  const res = await fetch(`${API_BASE}/api/mcp/status`);
  if (!res.ok) throw new Error(`MCP status API failed: ${res.status}`);
  const data = await res.json();
  return Array.isArray(data) ? data : data.servers || [];
}

async function parseError(res, fallback) {
  let msg = `${fallback}: ${res.status}`;
  try {
    const data = await res.json();
    if (data?.detail) {
      msg = typeof data.detail === 'string'
        ? data.detail
        : JSON.stringify(data.detail);
    }
  } catch {
    /* ignore */
  }
  return new Error(msg);
}

export function MCPProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const refresh = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', loading: true });
    try {
      const servers = await fetchMcpServers();
      dispatch({ type: 'SET_SERVERS', servers });
    } catch (err) {
      dispatch({ type: 'SET_ERROR', error: err.message });
    }
  }, []);

  const reload = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/mcp/reload`, { method: 'POST' });
      if (!res.ok) throw new Error(`reload failed: ${res.status}`);
    } catch (err) {
      console.warn('[dsh] MCP reload 失败，回退到普通刷新:', err);
    }
    await refresh();
  }, [refresh]);

  /** URL 导入（原有） */
  const addServer = useCallback(async (url) => {
    const res = await fetch(`${API_BASE}/api/mcp/servers`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) throw await parseError(res, '添加失败');
    return res.json();
  }, []);

  /** 通用配置导入（新增） */
  const addServerConfig = useCallback(async (config) => {
    const res = await fetch(`${API_BASE}/api/mcp/servers/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) throw await parseError(res, '添加失败');
    return res.json();
  }, []);

  const removeServer = useCallback(async (name) => {
    const res = await fetch(
      `${API_BASE}/api/mcp/servers/${encodeURIComponent(name)}`,
      { method: 'DELETE' }
    );
    if (!res.ok) throw await parseError(res, '移除失败');
    return res.json();
  }, []);

  const toggleServer = useCallback(async (name, enabled) => {
    const res = await fetch(
      `${API_BASE}/api/mcp/servers/${encodeURIComponent(name)}/toggle`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          typeof enabled === 'boolean' ? { enabled } : {}
        ),
      }
    );
    if (!res.ok) throw await parseError(res, '切换失败');
    return res.json();
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  return (
    <MCPContext.Provider
      value={{
        state,
        dispatch,
        refresh,
        reload,
        addServer,
        addServerConfig,
        removeServer,
        toggleServer,
      }}
    >
      {children}
    </MCPContext.Provider>
  );
}