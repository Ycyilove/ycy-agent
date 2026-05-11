const API_BASE_URL = 'http://localhost:8000';

export const chatWithAI = async (message, history = []) => {
  const formData = new FormData();
  formData.append('message', message);
  formData.append('history', JSON.stringify(history));

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  return {
    textStream: async function* () {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              yield data.text;
            } catch { /* empty */ }
          }
        }
      }
    }
  };
};

export const chatWithRAG = async (message, history = [], useRag = false, mode = 'quick') => {
  const formData = new FormData();
  formData.append('message', message);
  formData.append('history', JSON.stringify(history));
  formData.append('use_rag', useRag);
  formData.append('mode', mode);

  const response = await fetch(`${API_BASE_URL}/api/chat/rag`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  return {
    textStream: async function* () {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              yield data.text;
            } catch { /* empty */ }
          }
        }
      }
    }
  };
};

export const analyzeWithFiles = async (message, files = []) => {
  const formData = new FormData();
  formData.append('message', message);

  for (const file of files) {
    formData.append('files', file);
  }

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  return {
    textStream: async function* () {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              yield data.text;
            } catch { /* empty */ }
          }
        }
      }
    }
  };
};

export const ragAddFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/rag/add_file`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const ragAddFiles = async (files) => {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }

  const response = await fetch(`${API_BASE_URL}/api/rag/add_files`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const ragGetStats = async () => {
  const response = await fetch(`${API_BASE_URL}/api/rag/stats`);

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const ragSearch = async (query, topK = 3) => {
  const formData = new FormData();
  formData.append('query', query);
  formData.append('top_k', topK);

  const response = await fetch(`${API_BASE_URL}/api/rag/search`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const ragQuery = async (query, topK = 3) => {
  const formData = new FormData();
  formData.append('query', query);
  formData.append('top_k', topK);

  const response = await fetch(`${API_BASE_URL}/api/rag/query`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const ragQueryStream = async (query, topK = 3) => {
  const formData = new FormData();
  formData.append('query', query);
  formData.append('top_k', topK);

  const response = await fetch(`${API_BASE_URL}/api/rag/query/stream`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  return {
    textStream: async function* () {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              yield data.text;
            } catch { /* empty */ }
          }
        }
      }
    }
  };
};

export const ragDeleteFile = async (filename) => {
  const formData = new FormData();
  formData.append('filename', filename);

  const response = await fetch(`${API_BASE_URL}/api/rag/file`, {
    method: 'DELETE',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const ragClearAll = async () => {
  const response = await fetch(`${API_BASE_URL}/api/rag/clear`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};


export const getToolsList = async () => {
  const response = await fetch(`${API_BASE_URL}/api/tools`);

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const getToolInfo = async (toolName) => {
  const response = await fetch(`${API_BASE_URL}/api/tools/${toolName}`);

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const analyzeIntent = async (message) => {
  const formData = new FormData();
  formData.append('message', message);

  const response = await fetch(`${API_BASE_URL}/api/tools/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const executeTool = async (toolName, parameters, confirmed = false) => {
  const formData = new FormData();
  formData.append('tool_name', toolName);
  formData.append('parameters', JSON.stringify(parameters));
  formData.append('confirmed', confirmed);

  const response = await fetch(`${API_BASE_URL}/api/tools/execute`, {
    method: 'POST',
    body: formData,
  });

  const contentType = response.headers.get('Content-Type') || '';

  // 检测是否是文件下载响应（text/csv 或 text/plain 认为是文件下载）
  if (contentType.includes('text/csv') || contentType.includes('text/plain')) {
    const blob = await response.blob();
    let filename = 'export.csv';

    // 尝试从 Content-Disposition 获取文件名
    const contentDisposition = response.headers.get('Content-Disposition') || '';
    const match = contentDisposition.match(/filename="?([^"]+)"?/);
    if (match) {
      filename = match[1];
    }

    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    return { success: true, downloaded: true, filename };
  }

  // 尝试解析为 JSON
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (e) {
    return { success: true, result: text };
  }
};


export const createSession = async (name = '') => {
  const formData = new FormData();
  formData.append('name', name);

  const response = await fetch(`${API_BASE_URL}/api/session/create`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const listSessions = async () => {
  const response = await fetch(`${API_BASE_URL}/api/session/list`);

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const getSession = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}`);

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const deleteSession = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const renameSession = async (sessionId, name) => {
  const formData = new FormData();
  formData.append('name', name);

  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}/rename`, {
    method: 'PUT',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const getSessionMessages = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}/messages`);

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const getSessionContext = async (sessionId, maxMessages = 20) => {
  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}/context?max_messages=${maxMessages}`);

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const addSessionMessage = async (sessionId, role, content) => {
  const formData = new FormData();
  formData.append('role', role);
  formData.append('content', content);

  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}/message`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const getSessionFiles = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}/files`);

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const addSessionFile = async (sessionId, fileName, filePath = null, fileType = null) => {
  const formData = new FormData();
  formData.append('file_name', fileName);
  if (filePath) formData.append('file_path', filePath);
  if (fileType) formData.append('file_type', fileType);

  const response = await fetch(`${API_BASE_URL}/api/session/${sessionId}/file`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const deleteSessionFile = async (fileId) => {
  const response = await fetch(`${API_BASE_URL}/api/session/file/${fileId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const searchSessionMessages = async (sessionId, keyword, limit = 10) => {
  const formData = new FormData();
  formData.append('session_id', sessionId);
  formData.append('keyword', keyword);
  formData.append('limit', limit);

  const response = await fetch(`${API_BASE_URL}/api/session/search`, {
    method: 'GET',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};

export const getModelInfo = async () => {
  const response = await fetch(`${API_BASE_URL}/api/model/info`);

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  return response.json();
};
