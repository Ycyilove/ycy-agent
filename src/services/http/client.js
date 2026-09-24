export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const getErrorMessage = async (response) => {
  let detail = `请求失败（${response.status}）`;
  try {
    const payload = await response.json();
    detail = payload.detail || detail;
  } catch {
    // Keep the HTTP status when the backend did not return JSON.
  }
  return detail;
};

export const request = async (path, options = {}) => {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) throw new Error(await getErrorMessage(response));
  return response;
};

export const requestJson = async (path, options = {}) => {
  const response = await request(path, options);
  return response.json();
};

export const openSseStream = async (path, body, signal) => {
  const response = await request(path, { method: 'POST', body, signal });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  return {
    events: async function* () {
      let buffer = '';
      const parseFrame = (frame) => {
        const data = frame
          .split(/\r?\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trim())
          .join('\n');
        if (!data || data === '[DONE]') return null;
        try {
          return JSON.parse(data);
        } catch {
          return null;
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() || '';
        for (const frame of frames) {
          const event = parseFrame(frame);
          if (event) yield event;
        }
      }

      buffer += decoder.decode();
      if (buffer.trim()) {
        const event = parseFrame(buffer);
        if (event) yield event;
      }
    },
  };
};

export const openTextStream = async (path, body) => {
  const stream = await openSseStream(path, body);
  return {
    textStream: async function* () {
      for await (const event of stream.events()) {
        if (typeof event.text === 'string') yield event.text;
      }
    },
  };
};

export const formData = (entries = {}) => {
  const data = new FormData();
  for (const [key, value] of Object.entries(entries)) {
    if (value !== undefined && value !== null) data.append(key, value);
  }
  return data;
};
