import { useState, useEffect } from 'react';
import { getToolsList, analyzeIntent, executeTool } from '../services/api';

function ToolsPanel({ isOpen, onClose }) {
  const [tools, setTools] = useState([]);
  const [selectedTool, setSelectedTool] = useState(null);
  const [inputMessage, setInputMessage] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [executionResult, setExecutionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadTools();
    }
  }, [isOpen]);

  const loadTools = async () => {
    try {
      const data = await getToolsList();
      setTools(data.tools || []);
    } catch (error) {
      console.error('加载工具失败:', error);
    }
  };

  const handleAnalyze = async () => {
    if (!inputMessage.trim()) return;
    setLoading(true);
    setAnalysisResult(null);
    setExecutionResult(null);
    setSelectedTool(null);

    try {
      const result = await analyzeIntent(inputMessage);
      setAnalysisResult(result.analysis);
      if (result.analysis?.tool) {
        setSelectedTool(result.analysis.tool);
      }
    } catch (error) {
      console.error('分析失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleExecute = async () => {
    if (!selectedTool) return;
    setConfirming(true);

    try {
      const result = await executeTool(
        selectedTool.name,
        selectedTool.parameters,
        selectedTool.needs_confirmation
      );

      if (result.status === 'confirmation_needed') {
        const confirmed = window.confirm(result.message);
        if (confirmed) {
          const confirmResult = await executeTool(selectedTool.name, selectedTool.parameters, true);
          setExecutionResult(confirmResult);
        }
      } else {
        setExecutionResult(result);
      }
    } catch (error) {
      console.error('执行失败:', error);
    } finally {
      setConfirming(false);
    }
  };

  const groupedTools = tools.reduce((acc, tool) => {
    const category = tool.category || 'other';
    if (!acc[category]) acc[category] = [];
    acc[category].push(tool);
    return acc;
  }, {});

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl w-full max-w-3xl max-h-[85vh] overflow-hidden shadow-2xl border border-white/10">
        <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-white">AI Agent 工具</h2>
          <button onClick={onClose} className="text-white/60 hover:text-white">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(85vh-140px)]">
          <div className="mb-6">
            <label className="block mb-2 text-sm font-medium text-white/80">
              输入任务描述
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="例如：运行Python代码计算1+2+3，或读取CSV文件 data.csv..."
                className="flex-1 px-4 py-3 bg-white/10 border border-white/10 rounded-xl text-white placeholder-white/40 focus:outline-none focus:border-violet-500"
                onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
              />
              <button
                onClick={handleAnalyze}
                disabled={loading || !inputMessage.trim()}
                className="px-6 py-3 bg-violet-600 hover:bg-violet-500 text-white rounded-xl transition-colors disabled:opacity-50"
              >
                {loading ? '分析中...' : '分析'}
              </button>
            </div>
          </div>

          {analysisResult && (
            <div className="mb-6 p-4 bg-white/5 rounded-xl">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-sm font-medium text-white/60">意图分析结果</span>
                {analysisResult.needs_tool ? (
                  <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs rounded-full">
                    需要工具
                  </span>
                ) : (
                  <span className="px-2 py-0.5 bg-white/10 text-white/40 text-xs rounded-full">
                    普通对话
                  </span>
                )}
              </div>
              {analysisResult.needs_tool && analysisResult.tool ? (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-violet-400 font-medium">{analysisResult.tool.name}</span>
                    {analysisResult.tool.needs_confirmation && (
                      <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded-full">
                        需要确认
                      </span>
                    )}
                  </div>
                  <div className="text-sm text-white/60">
                    参数: {JSON.stringify(analysisResult.tool.parameters, null, 2)}
                  </div>
                  <div className="flex gap-2 mt-3">
                    <button
                      onClick={handleExecute}
                      disabled={confirming}
                      className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl transition-colors disabled:opacity-50"
                    >
                      {confirming ? '执行中...' : '执行工具'}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-white/40">未识别到工具调用</div>
              )}
            </div>
          )}

          {executionResult && (
            <div className="mb-6 p-4 bg-white/5 rounded-xl">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-sm font-medium text-white/60">执行结果</span>
                {executionResult.success !== false ? (
                  <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs rounded-full">
                    成功
                  </span>
                ) : (
                  <span className="px-2 py-0.5 bg-rose-500/20 text-rose-400 text-xs rounded-full">
                    失败
                  </span>
                )}
              </div>
              <pre className="text-sm text-white/80 whitespace-pre-wrap bg-black/30 p-3 rounded-lg max-h-60 overflow-y-auto">
                {executionResult.result || executionResult.error || JSON.stringify(executionResult, null, 2)}
              </pre>
            </div>
          )}

          <div className="mb-4">
            <h3 className="text-sm font-medium text-white/60 mb-3">可用工具 ({tools.length}个)</h3>
            {Object.entries(groupedTools).map(([category, categoryTools]) => (
              <div key={category} className="mb-4">
                <div className="text-xs font-medium text-violet-400 uppercase mb-2">{category}</div>
                <div className="grid grid-cols-2 gap-2">
                  {categoryTools.map((tool) => (
                    <div
                      key={tool.name}
                      className={`p-3 rounded-xl transition-colors cursor-pointer ${
                        selectedTool?.name === tool.name
                          ? 'bg-violet-500/20 border border-violet-500/30'
                          : 'bg-white/5 hover:bg-white/10'
                      }`}
                      onClick={() => {
                        setSelectedTool({ name: tool.name, parameters: {}, needs_confirmation: tool.danger_level === 'medium' });
                        setAnalysisResult({ needs_tool: true, tool: { name: tool.name, parameters: {}, needs_confirmation: tool.danger_level === 'medium' } });
                      }}
                    >
                      <div className="text-white font-medium text-sm">{tool.name}</div>
                      <div className="text-xs text-white/40 mt-1 line-clamp-2">{tool.description}</div>
                      <div className="flex gap-2 mt-2">
                        <span className={`text-xs px-1.5 py-0.5 rounded ${
                          tool.danger_level === 'safe' ? 'bg-emerald-500/20 text-emerald-400' :
                          tool.danger_level === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                          'bg-rose-500/20 text-rose-400'
                        }`}>
                          {tool.danger_level}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="px-6 py-4 border-t border-white/10">
          <button
            onClick={onClose}
            className="w-full px-6 py-2 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-colors"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  );
}

export default ToolsPanel;
