import { createContext, useContext } from 'react';

export const MCPContext = createContext(null);

export function useMCP() {
  const ctx = useContext(MCPContext);
  if (!ctx) throw new Error('useMCP must be used within MCPProvider');
  return ctx;
}