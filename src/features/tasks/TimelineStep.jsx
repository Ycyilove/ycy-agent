import { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import DiffPreview from './DiffPreview';
import ApprovalCard from './ApprovalCard';
import { useApp } from '../../app/state/AppContext';

// 后端 API 基地址——和 http/client.js 保持一致
// 用于把后端返回的相对 URL（如 /api/resources/xxx）拼成绝对 URL
const API_BASE =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

function resolveUrl(url) {
  if (!url) return '';
  if (/^https?:\/\//i.test(url)) return url;   // 已是绝对 URL
  if (url.startsWith('/')) return `${API_BASE}${url}`;
  return url;
}

const STATUS_COLOR = {
  running: 'text-[var(--accent)]',
  done: 'text-[var(--success)]',
  failed: 'text-[var(--danger)]',
  waiting: 'text-[var(--warning)]',
  pending: 'text-[var(--dim)]',
};

const STATUS_ICON = {
  running: I.play,
  done: I.check,
  failed: I.close,
  waiting: I.alert,
  pending: I.chevron,
};

const RESOURCE_ICON = {
  image: I.image,
  pdf: I.pdf,
  excel: I.excel,
  word: I.word,
  ppt: I.ppt,
  text: I.text,
  audio: I.audio,
  video: I.video,
  archive: I.zip,
  table: I.table,
  file: I.doc,
};

const RESOURCE_LABEL = {
  image: '图片',
  pdf: 'PDF',
  excel: 'Excel',
  word: 'Word',
  ppt: 'PPT',
  text: '文本',
  audio: '音频',
  video: '视频',
  archive: '压缩包',
  table: '表格',
  file: '文件',
};

function formatSize(bytes) {
  if (!bytes && bytes !== 0) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function ResourceIcon({ kind, className = 'h-4 w-4' }) {
  const d = RESOURCE_ICON[kind] || RESOURCE_ICON.file;
  return <Icon d={d} className={className} />;
}

function ThinkingBlock({ content, status }) {
  const [expanded, setExpanded] = useState(status === 'running');
  const userToggled = useRef(false);

  const toggle = () => {
    userToggled.current = true;
    setExpanded((v) => !v);
  };

  if (!content && status !== 'running') return null;

  return (
    <div className="mt-2 overflow-hidden rounded-md border border-[var(--border)] bg-[var(--surface)]">
      <button
        type="button"
        onClick={toggle}
        className="flex w-full items-center justify-between px-3 py-2 text-left transition-colors hover:bg-[var(--surface-2)]"
      >
        <div className="flex items-center gap-2">
          <Icon d={I.spark} className="h-3.5 w-3.5 text-[var(--dim)]" />
          <span className="text-[11px] text-[var(--dim)]">
            {status === 'running' ? '思考中…' : '思考过程'}
          </span>
          {status === 'running' && (
            <span className="flex items-center gap-1">
              <span className="h-1 w-1 animate-pulse rounded-full bg-[var(--accent)]" />
              <span
                className="h-1 w-1 animate-pulse rounded-full bg-[var(--accent)]"
                style={{ animationDelay: '150ms' }}
              />
              <span
                className="h-1 w-1 animate-pulse rounded-full bg-[var(--accent)]"
                style={{ animationDelay: '300ms' }}
              />
            </span>
          )}
        </div>
        <Icon
          d={I.chevron}
          className={`h-3.5 w-3.5 text-[var(--dim)] transition-transform duration-200 ${
            expanded ? 'rotate-180' : ''
          }`}
        />
      </button>
      <div
        className="overflow-hidden border-t border-[var(--border)] transition-all duration-300"
        style={{ maxHeight: expanded ? 400 : 0, opacity: expanded ? 1 : 0 }}
      >
        <div className="max-h-[400px] overflow-y-auto px-3 py-2.5">
          <pre className="whitespace-pre-wrap font-sans text-[11px] leading-relaxed text-[var(--muted)]">
            {content}
          </pre>
        </div>
      </div>
    </div>
  );
}

function AnswerBlock({ content, status }) {
  if (!content && status !== 'running') return null;

  return (
    <div className="animate-fade-in mt-2 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2.5">
      <div className="prose prose-sm prose-invert max-w-none text-[var(--text)]
        prose-headings:text-[var(--text)] prose-headings:font-semibold
        prose-p:text-[var(--text)] prose-p:leading-relaxed
        prose-strong:text-[var(--text)] prose-strong:font-semibold
        prose-code:text-[var(--accent)] prose-code:bg-[var(--surface-2)] prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-[13px] prose-code:before:content-none prose-code:after:content-none
        prose-pre:bg-[var(--bg)] prose-pre:border prose-pre:border-[var(--border)] prose-pre:rounded-md
        prose-ul:text-[var(--text)] prose-ol:text-[var(--text)] prose-li:text-[var(--text)]
        prose-blockquote:border-l-[var(--border)] prose-blockquote:text-[var(--muted)]
        prose-a:text-[var(--accent)] prose-a:no-underline hover:prose-a:underline">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
      {status === 'running' && (
        <span className="ml-0.5 inline-block h-4 w-1 animate-pulse bg-[var(--accent)] align-middle" />
      )}
    </div>
  );
}

function TableBlock({ table }) {
  const columns = Array.isArray(table.columns) ? table.columns : [];
  const rows = Array.isArray(table.rows) ? table.rows : [];
  if (!columns.length) return null;

  const visible = rows.slice(0, 50);

  return (
    <div className="overflow-hidden rounded-md border border-[var(--border)] bg-[var(--surface)]">
      <div className="max-h-[400px] overflow-auto">
        <table className="w-full border-collapse text-[11px]">
          <thead className="bg-[var(--surface-2)] sticky top-0">
            <tr>
              {columns.map((col) => (
                <th
                  key={col}
                  className="whitespace-nowrap border-b border-[var(--border)] px-2 py-1.5 text-left font-medium text-[var(--muted)]"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((row, i) => (
              <tr
                key={i}
                className="border-b border-[var(--border)] last:border-0"
              >
                {columns.map((col) => (
                  <td
                    key={col}
                    className="whitespace-nowrap px-2 py-1 text-[var(--text)]"
                  >
                    {row && row[col] !== undefined && row[col] !== null
                      ? String(row[col])
                      : ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > visible.length && (
        <div className="border-t border-[var(--border)] px-2 py-1 text-[10px] text-[var(--dim)]">
          仅显示前 {visible.length} 行，共 {rows.length} 行
        </div>
      )}
    </div>
  );
}

function ResourceBlock({ resources }) {
  if (!resources || resources.length === 0) return null;

  const tables = resources.filter((r) => r.kind === 'table');
  const images = resources.filter((r) => r.kind === 'image' || r.kind === 'chart');
  const pdfs = resources.filter((r) => r.kind === 'pdf');
  const audios = resources.filter((r) => r.kind === 'audio');
  const videos = resources.filter((r) => r.kind === 'video');
  const others = resources.filter(
    (r) => !['table', 'image', 'chart', 'pdf', 'audio', 'video'].includes(r.kind)
  );

  return (
    <div className="mt-2 space-y-2">
      {tables.map((t) => (
        <TableBlock key={t.id} table={t} />
      ))}

      {images.length > 0 && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {images.map((r) => (
            <a
              key={r.id}
              href={resolveUrl(r.url)}
              target="_blank"
              rel="noreferrer"
              className="group overflow-hidden rounded-md border border-[var(--border)] bg-[var(--surface)] transition-colors hover:border-[var(--accent)]/40"
            >
              <img
                src={resolveUrl(r.url)}
                alt={r.filename}
                loading="lazy"
                className="max-h-40 w-full bg-[var(--bg)] object-contain"
              />
              <div className="flex items-center gap-1.5 px-2 py-1.5">
                <ResourceIcon kind="image" className="h-3 w-3 text-[var(--dim)]" />
                <span className="truncate text-[11px] text-[var(--muted)] group-hover:text-[var(--text)]">
                  {r.filename}
                </span>
              </div>
            </a>
          ))}
        </div>
      )}

      {pdfs.map((r) => (
        <div
          key={r.id}
          className="overflow-hidden rounded-md border border-[var(--border)] bg-[var(--surface)]"
        >
          <div className="flex items-center justify-between border-b border-[var(--border)] px-3 py-1.5">
            <div className="flex min-w-0 items-center gap-2">
              <ResourceIcon kind="pdf" className="h-4 w-4 flex-shrink-0 text-[var(--danger)]" />
              <span className="truncate text-[11px] text-[var(--text)]">
                {r.filename}
              </span>
              {r.size && (
                <span className="text-[10px] text-[var(--dim)]">
                  {formatSize(r.size)}
                </span>
              )}
            </div>
            <a
              href={resolveUrl(r.url)}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-[11px] text-[var(--accent)] hover:underline"
            >
              <Icon d={I.download} className="h-3 w-3" />
              打开
            </a>
          </div>
          <embed
            src={resolveUrl(r.url)}
            type="application/pdf"
            className="h-[420px] w-full"
          />
        </div>
      ))}

      {audios.map((r) => (
        <div
          key={r.id}
          className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-2.5"
        >
          <div className="mb-2 flex items-center gap-2">
            <ResourceIcon kind="audio" className="h-4 w-4 text-[var(--muted)]" />
            <span className="truncate text-[11px] text-[var(--text)]">
              {r.filename}
            </span>
          </div>
          <audio controls src={resolveUrl(r.url)} className="w-full" />
        </div>
      ))}

      {videos.map((r) => (
        <div
          key={r.id}
          className="overflow-hidden rounded-md border border-[var(--border)] bg-[var(--surface)]"
        >
          <video controls src={resolveUrl(r.url)} className="max-h-[420px] w-full" />
          <div className="flex items-center gap-2 px-2.5 py-1.5">
            <ResourceIcon kind="video" className="h-4 w-4 text-[var(--muted)]" />
            <span className="truncate text-[11px] text-[var(--muted)]">
              {r.filename}
            </span>
          </div>
        </div>
      ))}

      {others.length > 0 && (
        <div className="space-y-1.5">
          {others.map((r) => (
            <a
              key={r.id}
              href={resolveUrl(r.url)}
              download={r.filename}
              className="flex items-center gap-2.5 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 transition-colors hover:border-[var(--accent)]/40"
            >
              <ResourceIcon
                kind={r.kind}
                className="h-5 w-5 flex-shrink-0 text-[var(--muted)]"
              />
              <div className="min-w-0 flex-1">
                <div className="truncate text-[12px] text-[var(--text)]">
                  {r.filename}
                </div>
                <div className="text-[10px] text-[var(--dim)]">
                  {RESOURCE_LABEL[r.kind] || r.kind}
                  {r.size ? ` · ${formatSize(r.size)}` : ''}
                </div>
              </div>
              <Icon d={I.download} className="h-4 w-4 text-[var(--accent)]" />
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

export default function TimelineStep({ step, taskId, isLast }) {
  const { state } = useApp();
  const [expanded, setExpanded] = useState(false);

  const statusColor = STATUS_COLOR[step.status] || 'text-[var(--dim)]';
  const statusIcon = STATUS_ICON[step.status] || I.chevron;
  const hasDetail = step.path || step.diff || step.log || step.approval;

  const approvalId = step.approval?.id;
  const approvalResolved = Boolean(
    approvalId && state.resolvedApprovalIds?.has(approvalId)
  );

  if (step.kind === 'user') {
    return (
      <div className="animate-fade-in flex gap-3">
        <div className="flex flex-col items-center pt-1.5">
          <div className="flex h-4 w-4 items-center justify-center text-[var(--muted)]">
            <Icon d={I.chat} className="h-3.5 w-3.5" />
          </div>
          {!isLast && <div className="mt-1 w-px flex-1 bg-[var(--border)]" />}
        </div>
        <div className="min-w-0 flex-1 pb-5">
          <div className="whitespace-pre-wrap rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2.5 text-sm text-[var(--text)]">
            {step.label}
          </div>
        </div>
      </div>
    );
  }

  if (step.kind === 'think') {
    return (
      <div className="animate-fade-in flex gap-3">
        <div className="flex flex-col items-center pt-1.5">
          <div className={`flex h-4 w-4 items-center justify-center ${statusColor}`}>
            <Icon d={statusIcon} className="h-3.5 w-3.5" />
          </div>
          {!isLast && <div className="mt-1 w-px flex-1 bg-[var(--border)]" />}
        </div>
        <div className="min-w-0 flex-1 pb-3">
          <div className="text-sm text-[var(--muted)]">{step.label}</div>
          {(step.log || step.status === 'running') && (
            <ThinkingBlock content={step.log} status={step.status} />
          )}
        </div>
      </div>
    );
  }

  if (step.kind === 'answer') {
    return (
      <div className="animate-fade-in flex gap-3">
        <div className="flex flex-col items-center pt-1.5">
          <div className={`flex h-4 w-4 items-center justify-center ${statusColor}`}>
            <Icon d={statusIcon} className="h-3.5 w-3.5" />
          </div>
          {!isLast && <div className="mt-1 w-px flex-1 bg-[var(--border)]" />}
        </div>
        <div className="min-w-0 flex-1 pb-5">
          <div className="text-sm font-medium text-[var(--text)]">{step.label}</div>
          <AnswerBlock content={step.log} status={step.status} />
        </div>
      </div>
    );
  }

  if (step.kind === 'resource') {
    return (
      <div className="animate-fade-in flex gap-3">
        <div className="flex flex-col items-center pt-1.5">
          <div className={`flex h-4 w-4 items-center justify-center ${statusColor}`}>
            <Icon d={statusIcon} className="h-3.5 w-3.5" />
          </div>
          {!isLast && <div className="mt-1 w-px flex-1 bg-[var(--border)]" />}
        </div>
        <div className="min-w-0 flex-1 pb-5">
          <div className="text-sm text-[var(--muted)]">{step.label}</div>
          <ResourceBlock resources={step.resources || []} />
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in flex gap-3">
      <div className="flex flex-col items-center pt-1.5">
        <div className={`flex h-4 w-4 items-center justify-center ${statusColor}`}>
          <Icon d={statusIcon} className="h-3.5 w-3.5" />
        </div>
        {!isLast && <div className="mt-1 w-px flex-1 bg-[var(--border)]" />}
      </div>

      <div className="min-w-0 flex-1 pb-5">
        <button
          type="button"
          onClick={() => hasDetail && setExpanded(!expanded)}
          className={`w-full text-left ${hasDetail ? 'cursor-pointer' : 'cursor-default'}`}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div
                className={`text-sm ${
                  step.status === 'pending' ? 'text-[var(--dim)]' : 'text-[var(--text)]'
                }`}
              >
                {step.label}
              </div>

              {step.path && (
                <div
                  className="mt-1 truncate font-mono text-[11px] text-[var(--muted)]"
                  title={step.path}
                >
                  {step.path}
                </div>
              )}

              {step.status === 'running' && (
                <div className="mt-1.5 flex items-center gap-1">
                  <span className="h-1 w-1 animate-pulse rounded-full bg-[var(--accent)]" />
                  <span
                    className="h-1 w-1 animate-pulse rounded-full bg-[var(--accent)]"
                    style={{ animationDelay: '150ms' }}
                  />
                  <span
                    className="h-1 w-1 animate-pulse rounded-full bg-[var(--accent)]"
                    style={{ animationDelay: '300ms' }}
                  />
                </div>
              )}
            </div>

            {hasDetail && (
              <Icon
                d={I.chevron}
                className={`mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-[var(--dim)] transition-transform duration-200 ${
                  expanded ? 'rotate-180' : ''
                }`}
              />
            )}
          </div>
        </button>

        {expanded && (
          <div className="mt-2 space-y-2">
            {step.diff && <DiffPreview diff={step.diff} />}
            {step.log && (
              <pre className="max-h-48 overflow-x-auto overflow-y-auto rounded-md border border-[var(--border)] bg-[var(--surface)] p-2.5 font-mono text-[11px] text-[var(--muted)]">
                {step.log}
              </pre>
            )}
            {step.approval && !approvalResolved && (
              <ApprovalCard
                approval={step.approval}
                taskId={taskId}
                stepId={step.id}
              />
            )}
          </div>
        )}

        {!expanded
          && step.approval
          && !approvalResolved
          && step.status === 'waiting' && (
          <div className="mt-2">
            <ApprovalCard
              approval={step.approval}
              taskId={taskId}
              stepId={step.id}
            />
          </div>
        )}
      </div>
    </div>
  );
}