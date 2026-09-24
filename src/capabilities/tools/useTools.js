import { useMemo } from 'react';
import { useTools } from './ToolsContext';
import { parseToolName, normalizeToolName } from '../../shared/capabilityUtils';

export function useToolsByIntent(intentTag) {
  const { state } = useTools();
  return useMemo(
    () => state.tools.filter((t) => (t.intent_tags || []).includes(intentTag)),
    [state.tools, intentTag]
  );
}

export function useToolsByServer(serverName) {
  const { state } = useTools();
  return useMemo(
    () =>
      state.tools.filter((t) => {
        const { server } = parseToolName(t.name);
        return server === serverName;
      }),
    [state.tools, serverName]
  );
}

export function useToolByName(fullName) {
  const { state } = useTools();
  return useMemo(() => {
    const normalized = normalizeToolName(fullName);
    return state.tools.find((t) => t.name === normalized) || null;
  }, [state.tools, fullName]);
}