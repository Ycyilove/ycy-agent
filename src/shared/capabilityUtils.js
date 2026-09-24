/**
 * 能力体系的公共工具函数。
 */

/**
 * 从完整工具名解析出 (server, tool)。
 * 例如 "mcp__filesystem__read_text_file" → { server: "filesystem", tool: "read_text_file" }
 *       "run_python_code" → { server: null, tool: "run_python_code" }
 */
export function parseToolName(fullName) {
  if (!fullName || typeof fullName !== 'string') {
    return { server: null, tool: '' };
  }
  if (!fullName.startsWith('mcp__')) {
    return { server: null, tool: fullName };
  }
  const parts = fullName.split('__');
  if (parts.length < 3) {
    return { server: null, tool: fullName };
  }
  // mcp__<server>__<tool>，tool 部分可能含 __
  const server = parts[1];
  const tool = parts.slice(2).join('__');
  return { server, tool };
}

/**
 * 判断一个工具是否来自指定 MCP server。
 */
export function isToolFromServer(fullName, serverName) {
  const { server } = parseToolName(fullName);
  return server === serverName;
}

/**
 * 规范化一个工具名（保证格式一致）。
 * 用于"前端旧格式 mcp:server:tool" → "mcp__server__tool" 的兼容。
 */
export function normalizeToolName(name) {
  if (!name || typeof name !== 'string') return '';
  // 旧格式：mcp:server:tool
  if (name.startsWith('mcp:') && !name.startsWith('mcp__')) {
    const parts = name.split(':');
    if (parts.length === 3) {
      return `mcp__${parts[1]}__${parts[2]}`;
    }
  }
  // 已是后端格式
  return name;
}