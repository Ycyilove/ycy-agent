import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { analyzeWithFiles, chatWithRAG, ragQueryStream, ragGetStats, createSession, addSessionMessage, getSessionMessages } from '../services/api';
import KnowledgeBasePanel from './KnowledgeBasePanel';
import SessionPanel from './SessionPanel';
import ToolsPanel from './ToolsPanel';

function ChatInterface() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isGlobalDragging, setIsGlobalDragging] = useState(false);
  const [dragFiles, setDragFiles] = useState([]);
  const [dragCounter, setDragCounter] = useState(0); // eslint-disable-line no-unused-vars
  const [useRagMode, setUseRagMode] = useState(false);
  const [answerMode, setAnswerMode] = useState('quick'); // 'quick' or 'deep'
  const [showModeDropdown, setShowModeDropdown] = useState(false);
  const [showAnswerDropdown, setShowAnswerDropdown] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState(null); // 当前会话ID
  const [knowledgeBaseOpen, setKnowledgeBaseOpen] = useState(false);
  const [sessionPanelOpen, setSessionPanelOpen] = useState(false);
  const [toolsPanelOpen, setToolsPanelOpen] = useState(false);
  const [hasKnowledgeBase, setHasKnowledgeBase] = useState(false);
  const [modelName, setModelName] = useState(''); // 当前模型名称
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    checkKnowledgeBase();
    fetchModelInfo();
  }, []);

  useEffect(() => {
    if (currentSessionId) {
      loadSessionMessages(currentSessionId);
    }
  }, [currentSessionId]);

  const loadSessionMessages = async (sessionId) => {
    if (!sessionId) return;
    try {
      const data = await getSessionMessages(sessionId);
      if (data.messages) {
        const loadedMessages = data.messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          mode: 'normal'
        }));
        setMessages(loadedMessages);
      }
    } catch (error) {
      console.error('加载会话消息失败:', error);
    }
  };

  const checkKnowledgeBase = async () => {
    try {
      const stats = await ragGetStats();
      setHasKnowledgeBase(stats.total_files > 0);
    } catch {
      setHasKnowledgeBase(false);
    }
  };

  const parseDeepThinkingContent = (content) => {
    const thinkingMatch = content.match(/思考：([\s\S]*?)(?=答案：|$)/);
    const answerMatch = content.match(/答案：([\s\S]*?)$/);

    return {
      thinking: thinkingMatch ? thinkingMatch[1].trim() : '',
      answer: answerMatch ? answerMatch[1].trim() : content
    };
  };

  const ThinkingMessage = ({ content }) => {
    const [isExpanded, setIsExpanded] = useState(true);
    const { thinking, answer } = parseDeepThinkingContent(content);

    if (!thinking) {
      return (
        <div className="prose prose-base max-w-none prose-headings:text-white prose-bold:text-white prose-strong:text-amber-300 prose-p:text-white/90 prose-ul:text-white/90 prose-ol:text-white/90 prose-li:text-white/90">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      );
    }

    return (
      <div className="space-y-3">
        <div className="border border-amber-500/30 rounded-xl overflow-hidden bg-gradient-to-br from-amber-900/20 to-amber-800/10">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full px-4 py-2 flex items-center justify-between bg-amber-500/10 hover:bg-amber-500/20 transition-colors"
          >
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              <span className="text-sm font-medium text-amber-300">思考过程</span>
            </div>
            <svg
              className={`w-4 h-4 text-amber-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {isExpanded && (
            <div className="p-4 border-t border-amber-500/20">
              <div className="prose prose-xs max-w-none prose-headings:text-amber-200 prose-bold:text-amber-200 prose-strong:text-amber-300 prose-p:text-white/70 prose-ul:text-white/70 prose-ol:text-white/70 prose-li:text-white/70 prose-blockquote:text-white/60">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{thinking}</ReactMarkdown>
              </div>
            </div>
          )}
        </div>
        <div className="prose prose-base max-w-none prose-headings:text-white prose-bold:text-white prose-strong:text-amber-300 prose-p:text-white/90 prose-ul:text-white/90 prose-ol:text-white/90 prose-li:text-white/90 prose-code:text-cyan-300 prose-pre:bg-slate-800/50 prose-blockquote:border-l-amber-400 prose-blockquote:text-white/80">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
        </div>
      </div>
    );
  };

  const fetchModelInfo = async () => {
    try {
      const info = await getModelInfo();
      if (info.display_name) {
        setModelName(info.display_name);
      }
    } catch {
      setModelName('');
    }
  };

  const handleFileSelect = (files) => {
    const validFiles = Array.from(files).filter(file => {
      const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'application/pdf'];
      return validTypes.includes(file.type);
    });
    setUploadedFiles(prev => [...prev, ...validFiles]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const handleGlobalDragOver = (e) => {
    e.preventDefault();
  };

  const handleGlobalDragEnter = (e) => {
    e.preventDefault();
    if (e.dataTransfer.types.includes('Files')) {
      setDragCounter(prev => prev + 1);
      setIsGlobalDragging(true);
      if (dragFiles.length === 0) {
        setDragFiles(Array.from(e.dataTransfer.files));
      }
    }
  };

  const handleGlobalDragLeave = (e) => {
    e.preventDefault();
    setDragCounter(prev => {
      const newCounter = prev - 1;
      if (newCounter <= 0) {
        setIsGlobalDragging(false);
        setDragFiles([]);
        return 0;
      }
      return newCounter;
    });
  };

  const handleGlobalDrop = (e) => {
    e.preventDefault();
    setDragCounter(0);
    setIsGlobalDragging(false);
    if (e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files);
    }
    setDragFiles([]);
  };

  const removeFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() && uploadedFiles.length === 0) return;

    // 自动保存会话：如果没有当前会话ID，则创建新会话
    let sessionId = currentSessionId;
    if (!sessionId) {
      try {
        const result = await createSession('未命名会话');
        sessionId = result.session_id;
        setCurrentSessionId(sessionId);
      } catch (error) {
        console.error('创建会话失败:', error);
      }
    }

    const userMessage = {
      role: 'user',
      content: input,
      files: uploadedFiles.map(f => ({ name: f.name, type: f.type })),
      mode: useRagMode ? 'knowledge' : 'normal'
    };
    setMessages(prev => [...prev, userMessage]);

    const currentInput = input;
    const currentFiles = [...uploadedFiles];
    const currentRagMode = useRagMode;
    const currentAnswerMode = answerMode;
    setInput('');
    setUploadedFiles([]);
    setIsLoading(true);

    try {
      // 保存用户消息到会话
      if (sessionId) {
        await addSessionMessage(sessionId, 'user', currentInput);
      }

      let stream;
      if (currentFiles.length > 0) {
        stream = await analyzeWithFiles(currentInput, currentFiles);
      } else if (currentRagMode && hasKnowledgeBase) {
        stream = await ragQueryStream(currentInput);
      } else {
        const history = messages.map(m => ({ role: m.role, content: m.content }));
        stream = await chatWithRAG(currentInput, history, currentRagMode && hasKnowledgeBase, currentAnswerMode);
      }

      let assistantMessage = { role: 'assistant', content: '', mode: currentRagMode && hasKnowledgeBase ? 'knowledge' : 'normal' };

      for await (const text of stream.textStream()) {
        assistantMessage.content += text;
        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          if (lastMessage.role === 'assistant') {
            return [...prev.slice(0, -1), { ...lastMessage, content: assistantMessage.content }];
          }
          return [...prev, { ...assistantMessage }];
        });
      }

      // 保存助手消息到会话
      if (sessionId && assistantMessage.content) {
        await addSessionMessage(sessionId, 'assistant', assistantMessage.content);
      }
    } catch (error) {
      console.error('Error chatting with AI:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: '抱歉，处理请求时发生错误，请检查后端服务是否启动。', mode: currentRagMode && hasKnowledgeBase ? 'knowledge' : 'normal' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleRagMode = () => {
    if (!hasKnowledgeBase) {
      setKnowledgeBaseOpen(true);
    } else {
      setUseRagMode(!useRagMode);
    }
  };

  const switchToNormalMode = () => {
    setUseRagMode(false);
  };

  const renderFilePreview = (file, index) => {
    const isImage = file.type.startsWith('image/');
    return (
      <div key={index} className="relative group">
        {isImage ? (
          <div className="relative">
            <img
              src={URL.createObjectURL(file)}
              alt={file.name}
              className="w-12 h-12 object-cover rounded-lg border-2 border-white/20 shadow-lg"
            />
            <button
              type="button"
              onClick={() => removeFile(index)}
              className="absolute -top-2 -right-2 w-5 h-5 bg-rose-500 rounded-full text-white flex items-center justify-center text-xs opacity-0 group-hover:opacity-100 transition-opacity hover:bg-rose-600"
            >
              ×
            </button>
          </div>
        ) : (
          <div className="relative">
            <div className="w-16 h-16 bg-gradient-to-br from-indigo-500/30 to-purple-500/30 rounded-xl border-2 border-white/20 flex items-center justify-center backdrop-blur-sm">
              <svg className="w-8 h-8 text-indigo-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <button
              type="button"
              onClick={() => removeFile(index)}
              className="absolute -top-2 -right-2 w-5 h-5 bg-rose-500 rounded-full text-white flex items-center justify-center text-xs opacity-0 group-hover:opacity-100 transition-opacity hover:bg-rose-600"
            >
              ×
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex flex-col"
      onDragOver={handleGlobalDragOver}
      onDragEnter={handleGlobalDragEnter}
      onDragLeave={handleGlobalDragLeave}
      onDrop={handleGlobalDrop}
    >
      <header className="px-6 py-4 border-b border-white/10 backdrop-blur-md bg-white/5">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-fuchsia-500 rounded-xl flex items-center justify-center shadow-lg shadow-violet-500/30">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-violet-300 to-fuchsia-300 bg-clip-text text-transparent">
                AI 对话助手
              </h1>
              <p className="text-xs text-white/40">Powered by {modelName || 'AI Model'}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSessionPanelOpen(true)}
              className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-xl text-sm font-medium text-white/80 transition-all"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              会话
            </button>
            <button
              onClick={() => setKnowledgeBaseOpen(true)}
              className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-xl text-sm font-medium text-white/80 transition-all"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
              </svg>
              知识库
            </button>
            <button
              onClick={() => setToolsPanelOpen(true)}
              className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-xl text-sm font-medium text-white/80 transition-all"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              工具
            </button>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
              <span className="text-sm text-white/60">在线</span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden flex flex-col max-w-4xl mx-auto w-full p-4">
        <div className="flex-1 overflow-y-auto space-y-6 pr-2 scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
          {messages.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center h-full text-center py-20">
              <div className="w-24 h-24 mb-6 bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 rounded-full flex items-center justify-center backdrop-blur-sm">
                <svg className="w-12 h-12 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-white/90 mb-2">开始对话</h2>
              <p className="text-white/50 max-w-md mb-4">
                发送消息开始对话，或上传图片/PDF文件进行分析
              </p>
              {!hasKnowledgeBase && (
                <button
                  onClick={() => setKnowledgeBaseOpen(true)}
                  className="px-6 py-3 bg-violet-600 hover:bg-violet-500 text-white rounded-xl font-medium transition-colors"
                >
                  导入知识库
                </button>
              )}
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={index}
                className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''} animate-fadeIn`}
              >
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  message.role === 'user'
                    ? 'bg-gradient-to-br from-blue-500 to-cyan-500'
                    : message.mode === 'knowledge'
                      ? 'bg-gradient-to-br from-violet-500 to-fuchsia-500'
                      : 'bg-gradient-to-br from-violet-500 to-fuchsia-500'
                }`}>
                  {message.role === 'user' ? (
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  )}
                </div>

                <div className={`flex flex-col max-w-[70%] ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
                  {message.files && message.files.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-2">
                      {message.files.map((file, fIndex) => (
                        <div key={fIndex} className="flex items-center gap-1.5 bg-white/10 backdrop-blur-sm px-3 py-1.5 rounded-full text-xs">
                          <svg className="w-3.5 h-3.5 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                          </svg>
                          <span className="text-white/80">{file.name}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {message.mode === 'knowledge' && (
                    <div className="flex items-center gap-1.5 bg-violet-500/20 backdrop-blur-sm px-2 py-1 rounded-full text-xs mb-2">
                      <svg className="w-3 h-3 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                      </svg>
                      <span className="text-violet-300">知识库问答</span>
                    </div>
                  )}
                  <div className={`rounded-2xl shadow-lg backdrop-blur-md transition-all duration-300 ${
                    message.role === 'user'
                      ? 'bg-gradient-to-br from-blue-600/90 to-cyan-600/90 text-white rounded-br-md px-5 py-4'
                      : 'bg-white/8 border border-white/15 text-white/95 rounded-bl-md px-5 py-4'
                  }`}>
                    {message.role === 'assistant' ? (
                      <ThinkingMessage content={message.content} />
                    ) : (
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}

          {isLoading && messages.length > 0 && messages[messages.length - 1].role !== 'assistant' && (
            <div className="flex gap-3 animate-fadeIn">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div className="bg-white/10 border border-white/10 rounded-2xl rounded-bl-md px-4 py-3 backdrop-blur-sm">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="mt-4 space-y-3">
          {uploadedFiles.length > 0 && (
            <div
              className={`rounded-xl p-3 transition-all duration-300 ${
                isDragging
                  ? 'bg-violet-500/20 border-2 border-violet-400'
                  : 'bg-white/5 border-2 border-transparent hover:border-white/10'
              } backdrop-blur-md max-h-32 overflow-y-auto`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="flex items-center gap-3 overflow-x-auto">
                <div className="flex items-center gap-2">
                  <div className="flex gap-2">
                    {uploadedFiles.map((file, index) => renderFilePreview(file, index))}
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="flex items-center justify-center gap-3 mb-3 py-2">
            <div className="relative">
              <button
                onClick={() => {
                  setShowModeDropdown(!showModeDropdown);
                  setShowAnswerDropdown(false);
                }}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-xl text-sm font-medium text-white/80 transition-all"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  {useRagMode && hasKnowledgeBase ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  )}
                </svg>
                <span>{useRagMode && hasKnowledgeBase ? '知识库问答' : '普通问答'}</span>
                <svg className={`w-4 h-4 transition-transform ${showModeDropdown ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {showModeDropdown && (
                <div className="absolute left-0 bottom-full mb-2 w-48 bg-slate-800/95 backdrop-blur-md rounded-xl border border-white/10 shadow-xl py-1 z-50">
                  <button
                    onClick={() => {
                      setUseRagMode(false);
                      setShowModeDropdown(false);
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-2 text-left text-sm transition-colors ${!useRagMode ? 'bg-violet-600/20 text-white' : 'text-white/70 hover:bg-white/10'}`}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    <div className="flex-1">
                      <div className="font-medium">普通问答</div>
                      <div className="text-xs text-white/50">大模型问答模式</div>
                    </div>
                    {!useRagMode && (
                      <svg className="w-4 h-4 text-violet-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </button>
                  <button
                    onClick={() => {
                      if (hasKnowledgeBase) {
                        setUseRagMode(true);
                        setShowModeDropdown(false);
                      } else {
                        setKnowledgeBaseOpen(true);
                        setShowModeDropdown(false);
                      }
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-2 text-left text-sm transition-colors ${useRagMode && hasKnowledgeBase ? 'bg-violet-600/20 text-white' : 'text-white/70 hover:bg-white/10'}`}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                    </svg>
                    <div className="flex-1">
                      <div className="font-medium">知识库问答</div>
                      <div className="text-xs text-white/50">{hasKnowledgeBase ? 'RAG知识库模式' : '请先导入知识库'}</div>
                    </div>
                    {useRagMode && hasKnowledgeBase && (
                      <svg className="w-4 h-4 text-violet-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </button>
                </div>
              )}
            </div>

            <div className="relative">
              <button
                onClick={() => {
                  setShowAnswerDropdown(!showAnswerDropdown);
                  setShowModeDropdown(false);
                }}
                className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-xl text-sm font-medium text-white/80 transition-all"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  {answerMode === 'quick' ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  )}
                </svg>
                <span>{answerMode === 'quick' ? '快速回答' : '深度思考'}</span>
                <svg className={`w-4 h-4 transition-transform ${showAnswerDropdown ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {showAnswerDropdown && (
                <div className="absolute left-0 bottom-full mb-2 w-48 bg-slate-800/95 backdrop-blur-md rounded-xl border border-white/10 shadow-xl py-1 z-50">
                  <button
                    onClick={() => {
                      setAnswerMode('quick');
                      setShowAnswerDropdown(false);
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-2 text-left text-sm transition-colors ${answerMode === 'quick' ? 'bg-emerald-600/20 text-white' : 'text-white/70 hover:bg-white/10'}`}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <div className="flex-1">
                      <div className="font-medium">快速回答</div>
                      <div className="text-xs text-white/50">适用于大部分情况</div>
                    </div>
                    {answerMode === 'quick' && (
                      <svg className="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </button>
                  <button
                    onClick={() => {
                      setAnswerMode('deep');
                      setShowAnswerDropdown(false);
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-2 text-left text-sm transition-colors ${answerMode === 'deep' ? 'bg-orange-600/20 text-white' : 'text-white/70 hover:bg-white/10'}`}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    <div className="flex-1">
                      <div className="font-medium">深度思考</div>
                      <div className="text-xs text-white/50">擅长解决更难的问题</div>
                    </div>
                    {answerMode === 'deep' && (
                      <svg className="w-4 h-4 text-orange-400" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>

          <form onSubmit={handleSubmit} className="flex gap-3">
            <div className="flex-1 relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={uploadedFiles.length > 0 ? "请描述您想对图片做什么..." : (useRagMode && hasKnowledgeBase ? "知识库问答模式..." : "输入消息...")}
                className="w-full px-5 py-4 pr-12 rounded-2xl bg-white/10 border-2 border-white/10 text-white placeholder-white/40 backdrop-blur-md focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-all"
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-white/40 hover:text-white/70 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                </svg>
              </button>
            </div>
            <button
              type="submit"
              className="px-6 py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-medium rounded-2xl transition-all hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-lg shadow-violet-500/30"
              disabled={isLoading || (!input.trim() && uploadedFiles.length === 0)}
            >
              {isLoading ? (
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              )}
            </button>
          </form>
        </div>
      </main>

      <KnowledgeBasePanel
        isOpen={knowledgeBaseOpen}
        onClose={() => {
          setKnowledgeBaseOpen(false);
          checkKnowledgeBase();
        }}
        onModeChange={setUseRagMode}
        currentMode={useRagMode ? 'rag' : 'normal'}
      />

      <SessionPanel
        isOpen={sessionPanelOpen}
        onClose={() => setSessionPanelOpen(false)}
        onSessionSelect={(id) => {
          setCurrentSessionId(id);
          setSessionPanelOpen(false);
        }}
        currentSessionId={currentSessionId}
      />

      <ToolsPanel
        isOpen={toolsPanelOpen}
        onClose={() => setToolsPanelOpen(false)}
      />

      {isGlobalDragging && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 backdrop-blur-sm"
          onDragOver={(e) => e.preventDefault()}
          onDragEnter={(e) => e.preventDefault()}
        >
          <div className="bg-white rounded-3xl p-8 shadow-2xl max-w-2xl w-full mx-4">
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-violet-500 to-fuchsia-500 rounded-full flex items-center justify-center">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-slate-800 mb-2">释放以添加文件</h3>
              <p className="text-slate-500 mb-6">文件将添加到对话输入区域</p>
              
              {dragFiles.length > 0 && (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    {dragFiles.slice(0, 4).map((file, index) => (
                      <div key={index} className="bg-slate-50 rounded-xl p-4 border border-slate-200">
                        <div className="flex items-center gap-3">
                          {file.type.startsWith('image/') ? (
                            <div className="w-12 h-12 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-lg flex items-center justify-center">
                              <svg className="w-6 h-6 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                              </svg>
                            </div>
                          ) : (
                            <div className="w-12 h-12 bg-gradient-to-br from-blue-100 to-cyan-100 rounded-lg flex items-center justify-center">
                              <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                              </svg>
                            </div>
                          )}
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-slate-700 truncate">{file.name}</p>
                            <p className="text-xs text-slate-500">
                              {file.size < 1024 ? `${file.size} B` : 
                               file.size < 1024 * 1024 ? `${(file.size / 1024).toFixed(1)} KB` : 
                               `${(file.size / (1024 * 1024)).toFixed(1)} MB`}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  {dragFiles.length > 4 && (
                    <p className="text-sm text-slate-500">还有 {dragFiles.length - 4} 个文件...</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ChatInterface;
