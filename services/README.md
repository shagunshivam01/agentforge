# Model Context Protocol (MCP) Servers

This folder contains example implementations of the Model Context Protocol (MCP) using `FastMCP`. The examples demonstrate how to create standalone tool servers and how to connect to them using a LangGraph-powered client.

## 📁 Files

- **`mathserver.py`**: A `FastMCP` server that runs locally using `stdio` transport. It exposes two basic tools: `add` and `mult` (multiply).
- **`weatherserver.py`**: A `FastMCP` server that runs via HTTP (`streamable-http` transport) on port `8000`. It exposes a mock `getWeather` tool.
- **`client.py`**: A `MultiServerMCPClient` that dynamically connects to both servers simultaneously. It binds the tools from both servers to a `ChatGroq` ReAct agent (`qwen-qwq-32b`) and executes queries.

## 🚀 How to Run

Because dependencies are managed at the root level, run these scripts from the project root (`agentforge/`):

1. **Start the HTTP Weather Server**:
   Open a terminal and run:
   ```bash
   python mcp-server/weatherserver.py
   ```
   *The server will start listening for MCP connections over HTTP.*

2. **Run the Client**:
   Open a separate terminal and run:
   ```bash
   python mcp-server/client.py
   ```
   *The client will automatically launch the `mathserver.py` via standard I/O (stdio) and connect to the running weather server, then ask the AI agent about the weather.*
