# AgentForge: LangGraph & Model Context Protocol (MCP) Examples

Welcome to **AgentForge**, a playground and repository for building agentic workflows, chatbots, and custom tool integrations using LangGraph and the Model Context Protocol (MCP).

---

## 📂 Project Structure

```bash
agentforge/
├── basic-chatbot/
│   ├── basicChatbot.ipynb     # Basic LangGraph chatbot flow
│   └── humanFeedback.ipynb    # Chatbot with human-in-the-loop verification via interrupts
├── mcp-server/
│   ├── mathserver.py          # FastMCP server with math tools (stdio transport)
│   ├── weatherserver.py       # FastMCP weather server (streamable-http transport)
│   └── client.py              # Multi-server MCP client invoking math & weather tools
├── pyproject.toml             # Project-wide pyproject metadata
└── requirements.txt           # Project-wide python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.12+ installed. It is recommended to use `uv` for package management, but standard `pip` works as well.

### 2. Environment Setup
Create a `.env` file in the root directory and add your API keys:

```ini
GROQ_API_KEY="your-groq-api-key"
TAVILY_API_KEY="your-tavily-api-key"
```

### 3. Installation
Install the project dependencies:

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

---

## 🤖 Modules Overview

### 1. Basic Chatbots (`/basic-chatbot`)
This directory contains Jupyter notebooks that walk through the basics of state preservation and human interaction with LangGraph:
* **`basicChatbot.ipynb`**: Demonstrates setting up a StateGraph, binding search tools, and managing chat state.
* **`humanFeedback.ipynb`**: Highlights the use of LangGraph's `interrupt()` state and `Command(resume=...)` to pause execution and solicit human guidance before executing sensitive tools.

### 2. Model Context Protocol Servers (`/mcp-server`)
The Model Context Protocol (MCP) is an open standard that enables developers to build secure, two-way connections between data sources (servers) and AI applications (clients).

* **Math Server (`mathserver.py`)**: Runs locally using stdio transport. Exposes math tools (`add`, `mult`).
* **Weather Server (`weatherserver.py`)**: Runs as an HTTP streamable server. Exposes weather retrieval.
* **Client (`client.py`)**: Establishes a `MultiServerMCPClient` connection to both the math and weather servers, retrieves their tools dynamically, and executes a LangGraph ReAct agent using `ChatGroq` (specifically the `qwen-qwq-32b` model).

#### Running the MCP Example:
1. Start the HTTP weather server:
   ```bash
   python mcp-server/weatherserver.py
   ```
2. In a separate terminal, execute the client (the client will automatically launch the math server via `stdio`):
   ```bash
   python mcp-server/client.py
   ```

---

## 🛠️ Tech Stack
* **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
* **Model Integration**: [LangChain Groq](https://github.com/langchain-ai/langchain-groq)
* **Tools**: [LangChain Tavily Search](https://github.com/langchain-ai/langchain-tavily), Custom FastMCP Servers
* **Package Management**: `uv` / `pip`
