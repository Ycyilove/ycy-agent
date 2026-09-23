import { formData, requestJson } from '../http/client';

export const getModelInfo = () => requestJson('/api/model/info');
export const downloadModel = (modelName, localDir = null) => requestJson('/api/model/download', {
  method: 'POST',
  body: formData({ model_name: modelName, local_dir: localDir }),
});
export const switchModelSource = (source) => requestJson('/api/model/switch', {
  method: 'POST',
  body: formData({ source }),
});
