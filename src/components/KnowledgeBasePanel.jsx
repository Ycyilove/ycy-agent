import { useState, useEffect } from 'react';
import { ragAddFiles, ragGetStats, ragDeleteFile, ragClearAll } from '../services/api';

function KnowledgeBasePanel({ isOpen, onClose, currentMode }) {
  const isRagMode = currentMode === 'rag';
  const [stats, setStats] = useState({ total_chunks: 0, total_files: 0, files: [] });
  const [loading, setLoading] = useState(false);
  const [uploadStage, setUploadStage] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadStats();
    }
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
    setUploadStage('正在上传文件...');

    try {
      setUploadStage(`正在处理 ${files.length} 个文件...`);
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
    setLoading(true);
    setUploadStage('正在删除文件...');
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
    setUploadStage('正在清空知识库...');
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
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl w-full max-w-2xl max-h-[80vh] overflow-hidden shadow-2xl border border-white/10">
        <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">知识库管理</h2>
          <button
            onClick={onClose}
            className="text-white/60 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(80vh-140px)]">
          <div className="mb-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-4">
                <div className="bg-violet-500/20 px-4 py-2 rounded-xl">
                  <span className="text-violet-300 text-sm">文档数量</span>
                  <p className="text-2xl font-bold text-white">{stats.total_files}</p>
                </div>
                <div className="bg-emerald-500/20 px-4 py-2 rounded-xl">
                  <span className="text-emerald-300 text-sm">文本块</span>
                  <p className="text-2xl font-bold text-white">{stats.total_chunks}</p>
                </div>
              </div>
              <button
                onClick={handleClearAll}
                disabled={loading || stats.total_files === 0}
                className="px-4 py-2 bg-rose-500/20 hover:bg-rose-500/30 text-rose-400 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                清空知识库
              </button>
            </div>
          </div>

          <div className="mb-6">
            <label className="block mb-2 text-sm font-medium text-white/80">
              上传文档（支持 PDF、TXT、DOCX）
            </label>
            <input
              type="file"
              multiple
              accept=".pdf,.txt,.docx"
              onChange={handleFileUpload}
              disabled={isUploading}
              className="block w-full text-sm text-white/60 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-violet-500 file:text-white hover:file:bg-violet-600 cursor-pointer disabled:opacity-50"
            />
            {isUploading && (
              <div className="mt-2 flex items-center gap-2 text-violet-300">
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>{uploadStage}</span>
              </div>
            )}

            {uploadStage && !isUploading && (
              <div className="mt-2 p-3 bg-white/5 rounded-xl text-sm text-white/80">
                {uploadStage}
              </div>
            )}
          </div>

          {stats.files && stats.files.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-white/60 mb-3">已加载的文档</h3>
              <div className="space-y-2">
                {stats.files.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-white/5 rounded-xl hover:bg-white/10 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <svg className="w-5 h-5 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <span className="text-white">{file.filename}</span>
                      <span className="text-xs text-white/40">{file.chunk_count} 个块</span>
                    </div>
                    <button
                      onClick={() => handleDeleteFile(file.filename)}
                      disabled={loading}
                      className="p-2 text-rose-400 hover:bg-rose-500/20 rounded-lg transition-colors disabled:opacity-50"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {stats.total_files === 0 && (
            <div className="text-center py-12 text-white/40">
              <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p>知识库为空，请上传文档</p>
            </div>
          )}
        </div>

        <div className="px-6 py-4 border-t border-white/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm text-white/60">对话模式:</span>
              <span className={`text-sm font-medium ${isRagMode ? 'text-violet-400' : 'text-white/40'}`}>
                {isRagMode ? '知识库问答' : '普通对话'}
              </span>
            </div>
            <button
              onClick={onClose}
              className="px-6 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-xl transition-colors"
            >
              完成
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default KnowledgeBasePanel;
