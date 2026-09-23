import { createContext, useContext } from 'react';

export const ToolsContext = createContext(null);

export function useTools() {
  const ctx = useContext(ToolsContext);
  if (!ctx) throw new Error('useTools must be used within ToolsProvider');
  return ctx;
}