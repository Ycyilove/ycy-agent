import { formData, request, requestJson } from '../http/client';

export const getToolsList = () => requestJson('/api/tools');
export const getToolInfo = (toolName) => requestJson(`/api/tools/${toolName}`);
export const analyzeIntent = (message) => requestJson('/api/tools/analyze', {
  method: 'POST',
  body: formData({ message }),
});

export const executeTool = async (toolName, parameters, confirmed = false) => {
  const response = await request('/api/tools/execute', {
    method: 'POST',
    body: formData({
      tool_name: toolName,
      parameters: JSON.stringify(parameters),
      confirmed,
    }),
  });
  const contentType = response.headers.get('Content-Type') || '';

  if (contentType.includes('text/csv') || contentType.includes('text/plain')) {
    const blob = await response.blob();
    let filename = 'export.csv';
    const contentDisposition = response.headers.get('Content-Disposition') || '';
    const match = contentDisposition.match(/filename="?([^"]+)"?/);
    if (match) filename = match[1];

    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.URL.revokeObjectURL(url);
    return { success: true, downloaded: true, filename };
  }

  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return { success: true, result: text };
  }
};
