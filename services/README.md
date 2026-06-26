# Model Context Protocol (MCP) Servers

This folder contains example implementations of the Model Context Protocol (MCP) using `FastMCP`. The examples demonstrate how to create standalone tool servers and how to connect to them using a LangGraph-powered client.

## Files

- **`math/server.py`**: A `FastMCP` server that runs locally using `stdio` transport. It exposes two basic tools: `add` and `mult` (multiply).
- **`weather/server.py`**: A `FastMCP` server that runs via HTTP (`streamable-http` transport) on port `8000`. It exposes a mock `getWeather` tool.
- **`tavily/server.py`**: A `FastMCP` server that runs via HTTP on port '8000'. It exposes a tavily_search tool. 

## How to Run

Because dependencies are managed at the root level, run these scripts from the project root (`agentforge/`):

1. **Start the HTTP Weather Server**:
   Open a terminal and run:
   ```bash
   python services/mcp/{service-name}/server.py
   ```
