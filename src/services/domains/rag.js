import { API_BASE_URL, formData, request, requestJson } from '../http/client';

export const ragAddFile = (file) => requestJson('/api/rag/add_file', {
  method: 'POST',
  body: formData({ file }),
});

export const ragAddFiles = (files) => {
  const data = new FormData();
  for (const file of files) data.append('files', file);
  return requestJson('/api/rag/add_files', { method: 'POST', body: data });
};

export const ragGetStats = () => requestJson('/api/rag/stats');
export const ragSearch = (query, topK = 3) => requestJson('/api/rag/search', {
  method: 'POST',
  body: formData({ query, top_k: topK }),
});
export const ragQuery = (query, topK = 3) => requestJson('/api/rag/query', {
  method: 'POST',
  body: formData({ query, top_k: topK }),
});

export const ragQueryStream = async (query, topK = 3) => {
  const response = await request('/api/rag/query/stream', {
    method: 'POST',
    body: formData({ query, top_k: topK }),
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const streamState = { resources: [] };
  const normalizeResource = (resource) => {
    const url = resource.url?.startsWith('http')
      ? resource.url
      : `${API_BASE_URL}${resource.url}`;
    return {
      ...resource,
      url,
      markdown: resource.markdown?.replace(resource.url, url) || resource.markdown,
    };
  };

  return {
    get resources() {
      return streamState.resources;
    },
    textStream: async function* () {
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() || '';
        for (const frame of frames) {
          const line = frame.split(/\r?\n/).find((item) => item.startsWith('data: '));
          if (!line) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (Array.isArray(data.resources)) {
              streamState.resources = data.resources.map(normalizeResource);
            }
            if (typeof data.text === 'string' && data.text) yield data.text;
          } catch {
            // Ignore the SSE terminator and incomplete frames.
          }
        }
      }
    },
  };
};

export const ragDeleteFile = (filename) => requestJson('/api/rag/file', {
  method: 'DELETE',
  body: formData({ filename }),
});
export const ragClearAll = () => requestJson('/api/rag/clear', { method: 'DELETE' });
