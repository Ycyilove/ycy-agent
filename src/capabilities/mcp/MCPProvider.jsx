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

export function MCPProvider({ children, autoRefreshMs = 30000 }) {
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
    // 让后端重读 mcp_servers.json，同步新增/删除/变更
    try {
      const res = await fetch(`${API_BASE}/api/mcp/reload`, { method: 'POST' });
      if (!res.ok) throw new Error(`reload failed: ${res.status}`);
    } catch (err) {
      console.warn('[dsh] MCP reload 失败，回退到普通刷新:', err);
    }
    await refresh();
  }, [refresh]);

  const addServer = useCallback(async (url) => {
    const res = await fetch(`${API_BASE}/api/mcp/servers`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) {
      let msg = `添加失败: ${res.status}`;
      try {
        const data = await res.json();
        if (data?.detail) msg = data.detail;
      } catch {
        /* ignore */
      }
      throw new Error(msg);
    }
    return res.json();
  }, []);

  const removeServer = useCallback(async (name) => {
    const res = await fetch(
      `${API_BASE}/api/mcp/servers/${encodeURIComponent(name)}`,
      { method: 'DELETE' }
    );
    if (!res.ok) {
      let msg = `移除失败: ${res.status}`;
      try {
        const data = await res.json();
        if (data?.detail) msg = data.detail;
      } catch {
        /* ignore */
      }
      throw new Error(msg);
    }
    return res.json();
  }, []);

  useEffect(() => {
    // 首次用 reload：保证读的是最新文件
    reload();
    if (autoRefreshMs > 0) {
      const timer = setInterval(reload, autoRefreshMs);
      return () => clearInterval(timer);
    }
  }, [reload, autoRefreshMs]);

  return (
    <MCPContext.Provider
      value={{ state, dispatch, refresh, reload, addServer, removeServer }}
    >
      {children}
    </MCPContext.Provider>
  );
}