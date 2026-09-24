export default function DiffPreview({ diff }) {
  // diff: { path, hunks: [{ oldStart, lines: [{type:'ctx'|'add'|'del', text}] }] }
  return (
    <div className="border border-[var(--border)] rounded-md overflow-hidden bg-[var(--surface)]">
      <div className="px-2.5 py-1.5 border-b border-[var(--border)] font-mono text-[11px] text-[var(--muted)]">
        {diff.path}
      </div>
      <div className="font-mono text-[11px] leading-relaxed max-h-64 overflow-y-auto">
        {diff.hunks?.map((hunk, hi) => (
          <div key={hi}>
            {hunk.lines.map((line, li) => {
              const prefix = line.type === 'add' ? '+' : line.type === 'del' ? '-' : ' ';
              const color =
                line.type === 'add' ? 'text-[var(--success)] bg-[var(--success)]/5'
                : line.type === 'del' ? 'text-[var(--danger)] bg-[var(--danger)]/5'
                : 'text-[var(--muted)]';
              return (
                <div key={li} className={`px-2.5 whitespace-pre ${color}`}>
                  {prefix} {line.text}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}