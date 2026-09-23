export function adaptMCPServer(server) {
  return {
    name: server.name || server.id,
    transport: server.transport || 'stdio',
    status: server.status || 'configured',
    tools: server.tools || 0,
    url: server.url || null,
    error: server.error || null,
  };
}