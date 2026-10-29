import { formData, requestJson } from '../http/client';

export const getToolsList = () => requestJson('/api/tools');
export const getToolInfo = (toolName) => requestJson(`/api/tools/${toolName}`);
export const analyzeIntent = (message) => requestJson('/api/tools/analyze', {
  method: 'POST',
  body: formData({ message }),
});

// ⚠️ executeTool 已删除：
//    /api/tools/execute 路由因允许客户端 confirmed=true 绕过审批被移除。
//    所有工具执行必须走 streamAgentTask → orchestrator → SecurityGateway。
//    若未来需要"手动执行工具"功能，应新增：
//      POST /api/agent/tasks/{task_id}/tools/{tool_name}
//    由服务端做 danger_level + workspace 校验，不接受客户端 confirmed。