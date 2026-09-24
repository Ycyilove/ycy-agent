import { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import DiffPreview from './DiffPreview';
import ApprovalCard from './ApprovalCard';

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
  const images = resources.filter(
    (r) => r.kind === 'image' || r.kind === 'chart'
  );
  const files = resources.filter(
    (r) => r.kind !== 'table' && r.kind !== 'image' && r.kind !== 'chart'
  );

  return (
    <div className="mt-2 space-y-2">
      {tables.map((table) => (
        <TableBlock key={table.id} table={table} />
      ))}

      {(images.length > 0 || files.length > 0) && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {images.map((resource) => (
            <a
              key={resource.id}
              href={resource.url}
              target="_blank"
              rel="noreferrer"
              className="overflow-hidden rounded-md border border-[var(--border)] bg-[var(--surface)] transition-colors hover:border-[var(--accent)]/40"
            >
              <img
                src={resource.url}
                alt={resource.filename}
                className="max-h-32 w-full bg-[var(--bg)] object-contain"
              />
              <div className="truncate px-2 py-1.5 text-[11px] text-[var(--muted)]">
                {resource.filename}
              </div>
            </a>
          ))}
          {files.map((resource) => (
            <a
              key={resource.id}
              href={resource.url}
              download={resource.filename}
              className="flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--surface)] px-2.5 py-2 text-[11px] text-[var(--muted)] transition-colors hover:border-[var(--accent)]/40"
            >
              <Icon d={I.doc} className="h-3.5 w-3.5 flex-shrink-0" />
              <span className="truncate">{resource.filename}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

export default function TimelineStep({ step, taskId, isLast }) {
  const [expanded, setExpanded] = useState(false);

  const statusColor = STATUS_COLOR[step.status] || 'text-[var(--dim)]';
  const statusIcon = STATUS_ICON[step.status] || I.chevron;
  const hasDetail = step.path || step.diff || step.log || step.approval;

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

  // 默认：action 类型
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
            {step.approval && (
              <ApprovalCard
                approval={step.approval}
                taskId={taskId}
                stepId={step.id}
              />
            )}
          </div>
        )}

        {!expanded && step.approval && step.status === 'waiting' && (
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