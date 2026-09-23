/**
 * ToolsProvider —— 提供"当前可用的工具列表"。
 *
 * 数据来自 /api/tools。
 * 只读，不在这里启用/禁用。
 */

import { useEffect, useReducer, useCallback } from 'react';
import { ToolsContext } from './ToolsContext';
import { getToolsList } from '../../services/domains/tools';

const initialState = {
  tools: [],
  loading: false,
  error: null,
  lastFetchedAt: null,
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_TOOLS':
      return {
        ...state,
        tools: action.tools,
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

export function ToolsProvider({ children, autoRefreshMs = 60000 }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const refresh = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', loading: true });
    try {
      const data = await getToolsList();
      dispatch({ type: 'SET_TOOLS', tools: data.tools || [] });
    } catch (err) {
      dispatch({ type: 'SET_ERROR', error: err.message });
    }
  }, []);

  useEffect(() => {
    refresh();
    if (autoRefreshMs > 0) {
      const timer = setInterval(refresh, autoRefreshMs);
      return () => clearInterval(timer);
    }
  }, [refresh, autoRefreshMs]);

  return (
    <ToolsContext.Provider value={{ state, dispatch, refresh }}>
      {children}
    </ToolsContext.Provider>
  );
}