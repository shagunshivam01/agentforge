<!-- Badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/Package%20Manager-uv-de5fe9?style=flat-square&logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![Protocol: MCP](https://img.shields.io/badge/Protocol-MCP-orange?style=flat-square)](https://modelcontextprotocol.io/)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

# AgentForge MCP Chatbot

A modular **Model Context Protocol (MCP)** powered chatbot system with tool-use, planning, memory, and real-time agent execution.

It supports:
- Tool-based reasoning (MCP servers)
- LLM-based planning (Groq / Llama 3.1)
- Conversation memory
- Runtime execution engine
- Streamlit UI
- Extensible plugin architecture (future-ready)

---

## Features

- MCP Tool Registry (multi-server tool loading)
- LLM Planner (decides when to use tools)
- Runtime Execution Engine
- Conversation Memory
- Tavily Web Search integration
- Structured planning output (Pydantic-based)
- Tool normalization + safety layer
- Streamlit Chat UI

---

## Architecture

```bash
+-----------------------------------+
                  |           Streamlit UI            |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |          Memory Layer             |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |          LLM Planner              | <--- (Groq / Llama 3.1)
                  +-----------------+-----------------+
                                    |
                     +--------------+--------------+
                     |                             |
                     | (Tool Needed)               | (Direct Response)
                     v                             v
        +------------+------------+     +----------+----------+
        |     Runtime Engine      |     |     Synthesizer     |
        +------------+------------+     +----------+----------+
                     |                             ^
       +-------------+-------------+               |
       |  Normalized Tool Execution|               |
       v                           v               |
+------+------+             +------+------+        |
| Tavily Search|            | Other MCP   |        |
| (MCP Server) |            | Plugins     |        |
+------+------+             +------+------+        |
       |                           |               |
       +-------------+-------------+               |
                     |                             |
                     +---(Collects Tool Output)----+
```
---

## Core Execution Flow

AgentForge handles user queries deterministically via `MCPRuntime` utilizing a centralized orchestration loop:

```text
[User Input] ──> MCPRuntime.execute()
                     │
                     ├──> 1. ConversationMemory.add() (Persist query)
                     ├──> 2. MCPRegistry.get_tool_schemas() (Fetch capability definitions)
                     ├──> 3. MCPPlanner.plan() (LLM decides routing approach)
                     │
                     ├─── [Choice A: No Tools Needed]
                     │        └──> MCPPlanner.direct_response() ──> Synthesize Conversational Text
                     │
                     └─── [Choice B: Tool Execution Needed]
                              ├──> Loop through plan.tool_calls
                              ├──> MCPTelemetry.log_step()
                              ├──> tool.ainvoke() (Asynchronous tool call execution)
                              └──> MCPPlanner.synthesize() ──> Compile final data-driven answer
```

---

## Technical Design Highlights

- **Asynchronous Execution:** The entire workflow layer (`plan`, `direct_response`, `synthesize`) leverages Python `asyncio` to handle concurrent tool loading and high-throughput streaming workloads.
- **Strict Guardrailing:** Prompts explicitly decouple general knowledge, logic, and mathematics from tool usage, reducing token waste and avoiding agent loop degradation.
- **Deterministic Token Safety:** Using LangChain's structured orchestration guarantees your output parses safely into the underlying `PlanOutput` data model without risking loose markdown strings breaking JSON structures.

---

## Repository Structure

```text
.
├── experiments/                 # Jupyter notebooks for prototyping & research
│   └── basic-chatbot/
├── services/                    # Out-of-process MCP Server definitions
│   └── mcp/
│       ├── math/                # Math execution MCP server
│       ├── tavily/              # Search engine integration MCP server
│       └── weather/             # Live weather data MCP server
├── src/                         # Core AgentForge package
│   ├── apps/
│   │   └── chatbot/             # Chatbot UI application
│   │       ├── api/             # FastAPI / API server layers
│   │       └── ui/              # Streamlit frontend application
│   └── core/                    # Engine internals
│       ├── mcp/                 # Protocol, planner, tool registry, and runtime execution
│       ├── memory/              # Context tracking & conversation history layers
│       ├── types/               # Strong typing and schema definitions
│       └── config.py            # Global runtime configuration
├── pyproject.toml               # Project metadata and build dependencies
└── uv.lock                      # Deterministic lockfile managed by uv
```

---

## How It Works

### 1. User Input
User sends a message via Streamlit UI.

### 2. Memory Layer
Stores conversation history per user.

### 3. Planner (LLM)
Decides whether the current prompt requires external execution or direct text generation using LangChain's structured output bindings.

Output:
```json
{
  "requires_tools": true,
  "confidence": 0.85,
  "tool_calls": [
    {
      "name": "exact_tool_name",
      "arguments": {}
    }
  ],
  "reason": "..."
}
```

### 4. Runtime Engine
Normalizes tool input
Executes MCP tool
Collects result
Handles fallback errors

### 5. Synthesizer
Converts tool output into final natural response.

### MCP Tools
Tavily Search
Used for:
- Weather
- News
- Live data
- Real-time queries

### Planner Rules
- No tools for greetings
- No tools for math/general reasoning
- Tools only for real-time / external data

---

## Project Setup

### Prerequisites
- **Python:** 3.10 or higher
- **Package Manager:** `uv` (recommended) or `pip`
- [Groq API Key](https://console.groq.com/)
- [Tavily API Key](https://tavily.com/)

### Installation

```bash
git clone git@github.com:shagunshivam01/agentforge.git
cd agentforge

uv venv
source .venv/bin/activate

uv pip install -r requirements.txt
```

### Environment Variables

Create .env:
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here

### Run MCP Server (Tavily)

```bash
python services/mcp/tavily/server.py
```

### Run Chatbot UI

```bash
streamlit run src/apps/chatbot/ui/streamlit_app.py
```

### Example Queries

Direct response
- hello

Tool usage
- weather in san francisco
- Real-time data
- latest AI news

### Known Issues

- Planner may over-trigger tools in ambiguous queries
- MCP server loading errors may occur if endpoint is down
- Tool schema enforcement still evolving
- No persistent vector memory yet

The LLM should decide what to do, not do everything itself. Execution should be deterministic, safe, and modular.

---

## How to Add a New MCP Tool

### Extending the System: Adding New MCP Tools

AgentForge is built to be modular. To add a new tool:

1. **Create the Server:** Add a new directory under `src/services/mcp/[your_tool]/` and implement the MCP protocol interface.
2. **Register the Tool:** Add the tool definitions and schemas to the `MCP Tool Registry`.
3. **Update the Planner:** If necessary, update the Pydantic schema or prompt guidelines in the LLM Planner to ensure it knows when to invoke your new tool.

### Planner Schema

The planner strictly enforces the following Pydantic structure for deterministic routing:

```python
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class PlanOutput(BaseModel):
    requires_tools: bool = False
    confidence: float = 1.0
    reason: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
```

---

## Contributing

We welcome contributions to AgentForge! Whether you want to fix a bug, optimize the planner, or add a brand new MCP server plugin, here is how you can get involved:

### How to Contribute

1. **Fork the Repository:** Create your own fork of the project to your GitHub account.
2. **Create a Feature Branch:** 
   ```bash
   git checkout -b feature/amazing-new-mcp-tool
   ```
3. **Commit Your Changes:** Write clear, concise commit messages that explain why the change was made.
   ```bash
   git commit -m "feat: add filesystem mcp server plugin"
   ```
4. **Push to the Branch:**
    ```bash
    git push origin feature/amazing-new-mcp-tool
    ```
5. **Open a Pull Request:** Submit your PR against the main branch. Please provide a clear description of the changes and reference any related issues.

### Development Guidelines

- **Code Style:** We use ruff or black for formatting. Please ensure your code passes basic linting checks before submitting.
- **Type Hints:** AgentForge relies heavily on Pydantic and type validation. Ensure all new core functions include robust Python type hinting.
- **Testing:** If adding a new tool normalization layer or memory strategy, include corresponding unit tests under a tests/ directory.

---

## License

MIT

---

