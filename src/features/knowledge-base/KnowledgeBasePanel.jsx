import { useState, useEffect } from 'react';
import Icon from '../../shared/ui/Icon';
import { I } from '../../shared/ui/icons';
import { ragAddFiles, ragGetStats, ragDeleteFile, ragClearAll } from '../../services/api';

export default function KnowledgeBasePanel({ isOpen, onClose }) {
  const [stats, setStats] = useState({ total_chunks: 0, total_files: 0, files: [] });
  const [loading, setLoading] = useState(false);
  const [uploadStage, setUploadStage] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    if (isOpen) loadStats();
  }, [isOpen]);

  const loadStats = async () => {
    try {
      const data = await ragGetStats();
      setStats(data);
    } catch (error) {
      console.error('获取知识库统计失败:', error);
    }
  };

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    setIsUploading(true);
    setUploadStage('正在上传文件…');
    try {
      setUploadStage(`正在处理 ${files.length} 个文件…`);
      const result = await ragAddFiles(files);
      setUploadStage(result.message);
      await loadStats();
    } catch (error) {
      setUploadStage('上传失败: ' + error.message);
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  const handleDeleteFile = async (filename) => {
    if (!confirm(`确定删除「${filename}」？`)) return;
    setLoading(true);
    setUploadStage('正在删除文件…');
    try {
      const result = await ragDeleteFile(filename);
      setUploadStage(result.message);
      await loadStats();
    } catch (error) {
      setUploadStage('删除失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClearAll = async () => {
    if (!confirm('确定要清空全部知识库吗？此操作不可恢复！')) return;
    setLoading(true);
    setUploadStage('正在清空知识库…');
    try {
      const result = await ragClearAll();
      setUploadStage(result.message);
      await loadStats();
    } catch (error) {
      setUploadStage('清空失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="flex max-h-[80vh] w-full max-w-2xl flex-col overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
        {/* Header */}
        <div className="flex flex-shrink-0 items-center justify-between border-b border-[var(--border)] px-5 py-3">
          <h2 className="text-sm font-medium text-[var(--text)]">知识库管理</h2>
          <button
            onClick={onClose}
            className="rounded p-1 text-[var(--muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
          >
            <Icon d={I.close} className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5">
          {/* Stats */}
          <div className="mb-5 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2">
                <div className="text-[11px] text-[var(--muted)]">文档数量</div>
                <div className="text-lg font-semibold leading-tight text-[var(--text)]">
                  {stats.total_files}
                </div>
              </div>
              <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2">
                <div className="text-[11px] text-[var(--muted)]">文本块</div>
                <div className="text-lg font-semibold leading-tight text-[var(--text)]">
                  {stats.total_chunks}
                </div>
              </div>
            </div>
            <button
              onClick={handleClearAll}
              disabled={loading || stats.total_files === 0}
              className="rounded-md border border-[var(--border)] px-3 py-1.5 text-xs text-[var(--danger)] transition-colors hover:border-[var(--danger)]/40 hover:bg-[var(--danger)]/10 disabled:cursor-not-allowed disabled:opacity-40"
            >
              清空知识库
            </button>
          </div>

          {/* Upload */}
          <div className="mb-5">
            <label className="mb-2 block text-xs text-[var(--muted)]">
              上传文档（支持 PDF、TXT、DOCX）
            </label>
            <label className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-[var(--border)] px-4 py-6 transition-colors hover:border-[var(--accent)]/40 hover:bg-[var(--surface-2)]">
              <Icon d={I.folder} className="h-4 w-4 text-[var(--muted)]" />
              <span className="text-xs text-[var(--muted)]">点击选择文件，或拖拽到此处</span>
              <input
                type="file"
                multiple
                accept=".pdf,.txt,.docx"
                onChange={handleFileUpload}
                disabled={isUploading}
                className="hidden"
              />
            </label>

            {isUploading && (
              <div className="mt-2 flex items-center gap-2 text-xs text-[var(--muted)]">
                <svg className="h-3 w-3 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span>{uploadStage}</span>
              </div>
            )}
            {uploadStage && !isUploading && (
              <div className="mt-2 rounded-md border border-[var(--border)] bg-[var(--surface-2)] p-2.5 text-xs text-[var(--muted)]">
                {uploadStage}
              </div>
            )}
          </div>

          {/* File list */}
          {stats.files && stats.files.length > 0 && (
            <div>
              <h3 className="mb-2 text-xs text-[var(--muted)]">已加载的文档</h3>
              <div className="space-y-1.5">
                {stats.files.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between rounded-md border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2 transition-colors hover:border-[var(--border-hover)]"
                  >
                    <div className="flex min-w-0 items-center gap-2.5">
                      <Icon d={I.doc} className="h-4 w-4 flex-shrink-0 text-[var(--muted)]" />
                      <span className="truncate text-sm text-[var(--text)]">{file.filename}</span>
                      <span className="flex-shrink-0 text-[11px] text-[var(--dim)]">
                        {file.chunk_count} 块
                      </span>
                    </div>
                    <button
                      onClick={() => handleDeleteFile(file.filename)}
                      disabled={loading}
                      className="rounded p-1.5 text-[var(--dim)] transition-colors hover:bg-[var(--danger)]/10 hover:text-[var(--danger)] disabled:opacity-40"
                    >
                      <Icon d={I.trash} className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {stats.total_files === 0 && (
            <div className="py-12 text-center text-[var(--dim)]">
              <Icon d={I.doc} className="mx-auto mb-3 h-8 w-8 opacity-50" />
              <p className="text-xs">知识库为空，请上传文档</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex flex-shrink-0 items-center justify-end border-t border-[var(--border)] px-5 py-3">
          <button
            onClick={onClose}
            className="rounded-md bg-[var(--accent)] px-3 py-1.5 text-xs font-medium text-[var(--bg)] transition-colors hover:bg-[var(--accent-hover)]"
          >
            完成
          </button>
        </div>
      </div>
    </div>
  );
}
