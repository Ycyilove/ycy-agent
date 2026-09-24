import { useState, useMemo, useEffect } from 'react';
import { useTools } from './ToolsContext';
import { useMCP } from '../mcp/MCPContext';
import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { parseToolName } from '../../shared/capabilityUtils';

const AUTO_DISCOVER_KEY = 'dsh.autoDiscoverMcpTools';

export default function ToolsPanel({ isOpen, onClose }) {
  const { state: toolsState, refresh: refreshTools } = useTools();
  const {
    state: mcpState,
    reload: reloadMcp,
    addServer,
    removeServer,
  } = useMCP();

  const [search, setSearch] = useState('');
  const [importUrl, setImportUrl] = useState('');
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState('');
  const [refreshing, setRefreshing] = useState(false);

  const [autoDiscover, setAutoDiscover] = useState(() => {
    return localStorage.getItem(AUTO_DISCOVER_KEY) === 'true';
  });

  useEffect(() => {
    localStorage.setItem(AUTO_DISCOVER_KEY, String(autoDiscover));
  }, [autoDiscover]);

  const { builtin, mcpGroups, matchedCount } = useMemo(() => {
    const keyword = search.trim().toLowerCase();
    const match = (tool) => {
      if (!keyword) return true;
      const hay = `${tool.name} ${tool.display_name || ''} ${
        tool.description || ''
      }`.toLowerCase();
      return hay.includes(keyword);
    };

    const b = [];
    const groups = new Map();
    let count = 0;

    for (const tool of toolsState.tools || []) {
      if (!match(tool)) continue;
      count += 1;
      const { server } = parseToolName(tool.name);
      if (!server) {
        b.push(tool);
      } else {
        if (!groups.has(server)) groups.set(server, []);
        groups.get(server).push(tool);
      }
    }

    const sortedGroups = Array.from(groups.entries()).sort((a, b) =>
      a[0].localeCompare(b[0])
    );

    return { builtin: b, mcpGroups: sortedGroups, matchedCount: count };
  }, [toolsState.tools, search]);

  const mcpStatusByName = useMemo(() => {
    const map = new Map();
    for (const s of mcpState.servers || []) map.set(s.name, s);
    return map;
  }, [mcpState.servers]);

  const handleImport = async () => {
    const url = importUrl.trim();
    if (!url) return;
    setImporting(true);
    setImportError('');
    try {
      await addServer(url);
      setImportUrl('');
      await Promise.all([reloadMcp(), refreshTools()]);
    } catch (err) {
      setImportError(err.message || '导入失败');
    } finally {
      setImporting(false);
    }
  };

  const handleRemove = async (name) => {
    if (!window.confirm(`确定要移除 MCP「${name}」吗？`)) return;
    try {
      await removeServer(name);
      await Promise.all([reloadMcp(), refreshTools()]);
    } catch (err) {
      window.alert(err.message || '移除失败');
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await Promise.all([reloadMcp(), refreshTools()]);
    } finally {
      setRefreshing(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={onClose}
    >
      <div
        className="flex max-h-[85vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex flex-shrink-0 items-center justify-between border-b border-[var(--border)] px-5 py-3">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-medium text-[var(--text)]">工具</h2>
            <span className="text-[11px] text-[var(--dim)]">
              {matchedCount} / {toolsState.tools?.length ?? 0}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="rounded p-1 text-[var(--muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)] disabled:opacity-40"
              title="刷新工具与 MCP 状态"
            >
              <Icon
                d={I.restore}
                className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`}
              />
            </button>
            <button
              onClick={onClose}
              className="rounded p-1 text-[var(--muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
            >
              <Icon d={I.close} className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Toolbar */}
        <div className="flex-shrink-0 space-y-2.5 border-b border-[var(--border)] px-5 py-3">
          {/* 搜索 */}
          <div className="relative">
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="搜索工具名称或描述…"
              className="h-8 w-full rounded-md border border-[var(--border)] bg-[var(--surface-2)] pl-3 pr-8 text-xs text-[var(--text)] placeholder-[var(--dim)] transition-colors focus:border-[var(--accent)]/60 focus:outline-none"
            />
            {search && (
              <button
                type="button"
                onClick={() => setSearch('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-[var(--dim)] hover:text-[var(--text)]"
              >
                <Icon d={I.close} className="h-3 w-3" />
              </button>
            )}
          </div>

          {/* 自动搜索 MCP —— 强调开关 */}
          <label
            className={`flex cursor-pointer items-center gap-3 rounded-md border px-3 py-2.5 transition-colors ${
              autoDiscover
                ? 'border-[var(--accent)]/50 bg-[var(--accent)]/5'
                : 'border-[var(--border)] bg-[var(--surface-2)] hover:border-[var(--border-hover)]'
            }`}
          >
            <div
              className={`relative h-5 w-9 flex-shrink-0 rounded-full transition-colors ${
                autoDiscover ? 'bg-[var(--accent)]' : 'bg-[var(--border)]'
              }`}
            >
              <span
                className={`absolute top-0.5 block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${
                  autoDiscover ? 'left-4.5' : 'left-0.5'
                }`}
              />
            </div>
            <input
              type="checkbox"
              checked={autoDiscover}
              onChange={(e) => setAutoDiscover(e.target.checked)}
              className="hidden"
            />
            <div className="min-w-0 flex-1">
              <div className="text-xs font-medium text-[var(--text)]">
                自动搜索外部 MCP 工具
              </div>
              <div className="text-[10px] text-[var(--dim)]">
                {autoDiscover
                  ? '已开启：任务执行时会自动发现并使用外部 MCP 提供的工具'
                  : '已关闭：仅使用内置工具，不搜索外部 MCP'}
              </div>
            </div>
          </label>

          {/* 导入 MCP */}
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={importUrl}
              onChange={(e) => setImportUrl(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleImport()}
              placeholder="粘贴 MCP 地址（http:// 或 https://）"
              className="h-8 flex-1 rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-3 text-xs text-[var(--text)] placeholder-[var(--dim)] transition-colors focus:border-[var(--accent)]/60 focus:outline-none"
            />
            <button
              type="button"
              onClick={handleImport}
              disabled={!importUrl.trim() || importing}
              className="h-8 flex-shrink-0 rounded-md bg-[var(--accent)] px-3 text-xs font-medium text-[var(--bg)] transition-colors hover:bg-[var(--accent-hover)] disabled:cursor-not-allowed disabled:bg-[var(--border)] disabled:text-[var(--dim)]"
            >
              {importing ? '导入中…' : '导入'}
            </button>
          </div>

          {importError && (
            <div className="rounded-md border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-2 py-1 text-[11px] text-[var(--danger)]">
              {importError}
            </div>
          )}
        </div>

        {/* Body */}
        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          <ToolSection
            title="内置工具"
            count={builtin.length}
            emptyText={search ? '无匹配的内置工具' : '暂无内置工具'}
          >
            {builtin.map((tool) => (
              <ToolRow key={tool.name} tool={tool} />
            ))}
          </ToolSection>

          {mcpGroups.map(([serverName, tools]) => {
            const status = mcpStatusByName.get(serverName);
            return (
              <ToolSection
                key={serverName}
                title={serverName}
                count={tools.length}
                status={status}
                onRemove={() => handleRemove(serverName)}
                emptyText="无匹配工具"
              >
                {tools.map((tool) => {
                  const { tool: shortName } = parseToolName(tool.name);
                  return (
                    <ToolRow
                      key={tool.name}
                      tool={tool}
                      shortName={shortName}
                    />
                  );
                })}
              </ToolSection>
            );
          })}

          {matchedCount === 0 && (
            <div className="py-8 text-center text-xs text-[var(--dim)]">
              {search ? '没有匹配的工具' : '暂无可用工具'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const STATUS_STYLE = {
  connected: { color: 'text-[var(--success)]', label: '已连接' },
  error: { color: 'text-[var(--danger)]', label: '错误' },
  configured: { color: 'text-[var(--muted)]', label: '未连接' },
  disconnected: { color: 'text-[var(--dim)]', label: '已断开' },
};

function ToolSection({
  title,
  count,
  emptyText,
  status,
  onRemove,
  children,
}) {
  const statusInfo = status
    ? STATUS_STYLE[status.status] || {
        color: 'text-[var(--muted)]',
        label: status.status,
      }
    : null;

  return (
    <section>
      <div className="mb-1.5 flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--dim)]">
            {title}
          </span>
          <span className="text-[10px] text-[var(--dim)]">({count})</span>
          {statusInfo && (
            <span className={`text-[10px] ${statusInfo.color}`}>
              ● {statusInfo.label}
            </span>
          )}
        </div>
        {onRemove && (
          <button
            type="button"
            onClick={onRemove}
            className="rounded p-1 text-[var(--dim)] transition-colors hover:text-[var(--danger)]"
            title="移除此 MCP"
          >
            <Icon d={I.trash} className="h-3 w-3" />
          </button>
        )}
      </div>

      {count === 0 ? (
        <div className="col-span-2 py-3 text-center text-[11px] text-[var(--dim)]">
          {emptyText}
        </div>
      ) : (
        <ul className="grid grid-cols-2 gap-1.5">{children}</ul>
      )}

      {status?.error && (
        <div className="mt-1 break-all rounded border border-[var(--danger)]/30 bg-[var(--danger)]/10 px-2 py-1 text-[10px] text-[var(--danger)]">
          {status.error}
        </div>
      )}
    </section>
  );
}

function ToolRow({ tool, shortName }) {
  return (
    <li className="rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2">
      <div className="truncate text-xs font-medium text-[var(--text)]">
        {shortName || tool.display_name || tool.name}
      </div>
      {tool.description && (
        <div className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-[var(--dim)]">
          {tool.description}
        </div>
      )}
    </li>
  );
}