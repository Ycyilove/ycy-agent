import { AppProvider } from './state/AppStore';
import { MCPProvider } from '../capabilities/mcp/MCPProvider';
import { ToolsProvider } from '../capabilities/tools/ToolsProvider';
import AppShell from './shell/AppShell';
import '../styles/tokens.css';

function App() {
  return (
    <AppProvider>
      <MCPProvider>
        <ToolsProvider>
          <AppShell />
        </ToolsProvider>
      </MCPProvider>
    </AppProvider>
  );
}

export default App;