import { formData, openTextStream } from '../http/client';

export const chatWithAI = (message, history = []) => openTextStream(
  '/api/chat',
  formData({ message, history: JSON.stringify(history) }),
);

export const chatWithRAG = (message, history = [], useRag = false, mode = 'quick') => openTextStream(
  '/api/chat/rag',
  formData({ message, history: JSON.stringify(history), use_rag: useRag, mode }),
);

export const analyzeWithFiles = (message, files = []) => {
  const data = formData({ message });
  for (const file of files) data.append('files', file);
  return openTextStream('/api/analyze', data);
};
