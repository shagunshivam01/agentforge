# AgentForge

A modular Python framework for building and experimenting with LLM-powered agents.

AgentForge is designed around a small set of explicit abstractions — **agents, models, memory, tools, and execution state** — so that agent architectures and infrastructure can evolve independently.

The goal is not to hide agent execution behind a large framework, but to make the execution model explicit, replaceable, and easy to experiment with.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Ruff](https://img.shields.io/badge/code%20style-Ruff-000000?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/package%20manager-uv-de5fe9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

> **Status:** Early development. The Direct agent architecture, CLI, FastAPI backend, and React web application are currently working. ReAct and additional integrations are under active development.

---

## Why AgentForge?

Agent systems often combine several concerns in a single execution layer:

- agent architecture
- LLM providers
- tool execution
- conversation memory
- execution state
- external integrations
- application logic

This can make it difficult to change one part without affecting the others.

AgentForge separates these concerns so that:

- new agent architectures can be added independently
- model providers can be replaced
- memory implementations can be replaced
- tools can be local or externally provided
- MCP can be used as an integration rather than defining the agent architecture
- applications can interact with agents without knowing their internal execution strategy
- architectures can be tested independently

The guiding principle is:

> **Keep the framework modular, explicit, and easy to reason about.**

---

## Architecture

```text
                         APPLICATIONS
                    ┌─────────┼─────────┐
                    │         │         │
                    ▼         ▼         ▼
                  CLI      FastAPI     React
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                            AGENT
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
              ARCHITECTURE            MODEL
                    │
                    ▼
                  TOOLS
                    │
              ┌─────┴─────┐
              ▼           ▼
            LOCAL        MCP
            TOOLS      ADAPTER
                    │
                    ▼
                  MEMORY
```

The important boundary is that **agent architectures depend on framework abstractions, not on specific applications or integrations**.

For example:

```text
ReAct
  │
  ├── Model
  ├── Memory
  └── Tool
         ▲
         │
    ┌────┴────┐
    │         │
 Local Tool  MCP Tool
```

ReAct should not need to know whether a tool is implemented locally or exposed through MCP.

Similarly, a model implementation does not define the agent architecture.

---

## Core Concepts

### Agent

An `Agent` is the framework-level abstraction responsible for executing a task and producing a response.

```python
class Agent(ABC):

    @abstractmethod
    async def run(
        self,
        message: str,
        **kwargs: Any,
    ) -> str:
        ...
```

Applications interact with the `Agent` abstraction rather than depending on a specific architecture.

---

### Agent Architecture

An architecture defines **how an agent executes a task**.

Examples include:

```text
Direct
ReAct
Plan-and-Execute
Reflexion
...
```

Architectures live under:

```text
src/agentforge/agents/architectures/
```

This allows a new architecture to be introduced without modifying unrelated model, memory, tool, or MCP infrastructure.

---

### Direct

Direct is intentionally the simplest architecture.

```text
User
  │
  ▼
Agent
  │
  ▼
Model
  │
  ▼
Response
```

The Direct agent does not contain:

- a planning loop
- tool selection
- multi-step reasoning
- a complex state machine

This makes Direct a useful baseline for evaluating more sophisticated architectures.

**Current status:** Working end-to-end through the framework, CLI, FastAPI API, and React web application.

---

### ReAct

ReAct implements:

```text
Reason → Act → Observe
```

Conceptually:

```text
              ┌──────────────┐
              │     Input    │
              └──────┬───────┘
                     ▼
                  Reason
                     │
              Tool required?
                /         \
              no           yes
              │             │
              ▼             ▼
           Response      Tool Call
                            │
                            ▼
                        Observation
                            │
                            └──────► Reason
```

ReAct-specific execution logic belongs under:

```text
src/agentforge/agents/architectures/react/
```

The ReAct implementation should depend on generic `Model`, `Memory`, and `Tool` abstractions rather than directly depending on MCP, a specific model provider, or an external orchestration framework.

**Current status:** In development.

---

### Model

A `Model` represents an LLM or inference provider.

```python
class Model(ABC):

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        ...
```

Agent architectures depend on this abstraction instead of directly depending on a provider.

This allows implementations such as:

```text
Model
 ├── GroqModel
 ├── OpenAIModel
 ├── AnthropicModel
 └── LocalModel
```

to be substituted without changing the agent architecture.

---

### Memory

Memory stores and retrieves information used across conversations or executions.

```python
class Memory(ABC):

    @abstractmethod
    async def get_messages(self) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def add_message(
        self,
        role: str,
        content: str,
        **kwargs: Any,
    ) -> None:
        ...

    @abstractmethod
    async def clear(self) -> None:
        ...
```

The initial implementation provides conversation memory.

Future implementations can provide persistent or specialized storage without changing the agent architecture.

---

### State

State represents the structured state of an **ongoing agent execution**.

State is different from memory:

```text
Memory
  → information available across conversations/executions

State
  → information belonging to the current execution
```

For example, a ReAct execution may maintain:

```text
ReactState
├── messages
├── current action
├── observation
├── iteration
└── completion status
```

State should remain architecture-specific where appropriate.

---

### Tool

A `Tool` represents a capability that an agent can invoke.

```python
class Tool(ABC):

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        ...

    @abstractmethod
    async def invoke(
        self,
        **kwargs: Any,
    ) -> Any:
        ...
```

Tools are managed through a `ToolRegistry`.

```text
ToolRegistry
├── register
├── unregister
├── discover
├── definitions
└── invoke
```

The agent architecture interacts with the generic `Tool` abstraction rather than the implementation details of an individual tool.

---

## MCP

Model Context Protocol (MCP) is treated as an **integration mechanism**, not as an agent architecture.

The intended relationship is:

```text
Agent Architecture
        │
        ▼
   Tool Abstraction
        ▲
        │
   MCP Adapter
        │
        ▼
   MCP Servers
```

This means MCP should not determine whether an agent uses Direct, ReAct, Plan-and-Execute, or another architecture.

A tool can be provided by:

```text
Local implementation
        or
MCP server
```

while exposing the same framework-level tool contract to the agent.

The repository currently contains example MCP services under:

```text
services/mcp/
├── math/
├── tavily/
└── weather/
```

The framework-side MCP integration is kept separate from these services.

---

## Applications

AgentForge currently includes a chatbot application with multiple interfaces:

```text
apps/chatbot/
├── api/
│   ├── dependencies.py
│   ├── main.py
│   ├── routes.py
│   └── schemas.py
├── app.py
├── cli.py
└── runtime.py
```

### CLI

The CLI provides a terminal-based interface for interacting with the agent.

### FastAPI

The FastAPI application exposes the agent through HTTP APIs.

Interactive API documentation is available through FastAPI's generated documentation when the server is running.

### React

The React application provides a web-based chatbot interface:

```text
apps/chatbot-web/
├── src/
│   ├── App.jsx
│   ├── api.js
│   ├── index.css
│   └── main.jsx
└── ...
```

These applications consume the framework rather than defining agent architecture themselves.

---

## Repository Structure

```text
.
├── apps/
│   ├── chatbot/
│   │   ├── api/
│   │   ├── app.py
│   │   ├── cli.py
│   │   └── runtime.py
│   └── chatbot-web/
│       └── src/
│
├── services/
│   └── mcp/
│       ├── math/
│       ├── tavily/
│       └── weather/
│
├── experiments/
│   └── basic-chatbot/
│
├── src/
│   └── agentforge/
│       ├── agents/
│       │   ├── base.py
│       │   └── architectures/
│       │       ├── direct/
│       │       └── react/
│       │
│       ├── models/
│       │
│       ├── memory/
│       │
│       ├── tools/
│       │
│       ├── mcp/
│       │
│       └── infrastructure/
│
├── tests/
│   ├── agents/
│   ├── apps/
│   ├── memory/
│   └── tools/
│
├── pyproject.toml
├── README.md
└── uv.lock
```

### Package responsibilities

| Package | Responsibility |
|---|---|
| `agents/` | Agent contracts and execution architectures |
| `models/` | LLM/model abstractions and implementations |
| `memory/` | Conversation and persistent context |
| `tools/` | Capability abstractions and tool registry |
| `mcp/` | MCP protocol/runtime/integration |
| `infrastructure/` | Cross-cutting concerns |
| `apps/` | Application-specific interfaces and composition |
| `services/` | External services such as MCP servers |
| `tests/` | Automated tests |
| `experiments/` | Research and experimentation |

Applications should consume the framework rather than implement framework logic.

---

## Getting Started

AgentForge uses `uv` for dependency and environment management.

```bash
git clone <repository-url>
cd agentforge

uv sync --dev
```

Verify the environment:

```bash
uv run python --version
```

---

## Configuration

Provider-specific configuration is supplied through environment variables.

Create a local `.env` file based on your environment configuration.

For example:

```env
GROQ_API_KEY=your_groq_api_key
```

**Do not commit API keys or other secrets.**

---

## Running the Chatbot

### FastAPI backend

Run the API with:

```bash
uv run uvicorn apps.chatbot.api.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI's interactive documentation is available at:

```text
http://localhost:8000/docs
```

### CLI

The repository also includes a CLI interface for interacting with the chatbot.

Run it using the CLI entry point configured by the project.

### React frontend

From the web application directory:

```bash
cd apps/chatbot-web
npm install
npm run dev
```

The Vite development server will provide the local web interface.

---

## Testing

Run the complete test suite:

```bash
uv run python -m pytest
```

The current test suite covers core components including:

- Direct agent behavior
- FastAPI routes
- application runtime
- conversation memory
- tool registry

The framework should favor deterministic tests using fake implementations where possible.

For example:

```text
FakeModel
    │
    ▼
DirectAgent
    │
    ▼
Fake response
```

This allows agent behavior to be tested without making real LLM requests.

---

## Code Quality

AgentForge uses Ruff for linting.

Run:

```bash
uv run ruff check .
```

Automatically fix supported issues:

```bash
uv run ruff check . --fix
```

---

## Development Roadmap

AgentForge is being developed incrementally.

### Core

- [x] Agent abstraction
- [x] Model abstraction
- [x] Memory abstraction
- [x] Tool abstraction
- [x] Conversation memory
- [x] Tool registry
- [ ] Clean architecture boundaries
- [ ] Expanded deterministic core tests

### Agent Architectures

- [x] Direct agent
- [ ] ReAct agent
- [ ] Plan-and-Execute
- [ ] Reflexion
- [ ] Architecture selection

### Models

- [x] Model abstraction
- [x] Groq model implementation
- [ ] Additional interchangeable model providers

### Memory

- [x] Conversation memory
- [ ] Persistent storage
- [ ] Pluggable memory backends
- [ ] Context/window management
- [ ] Long-term memory

### Tools and MCP

- [x] Tool abstraction
- [x] Tool registry
- [ ] MCP tool adapter
- [ ] MCP tool discovery
- [ ] MCP tool execution

### Evaluation

- [ ] Evaluation datasets
- [ ] Architecture comparison
- [ ] Latency measurements
- [ ] Tool-use metrics
- [ ] Cost measurements
- [ ] Reproducible experiments

### Observability

- [ ] Agent execution tracing
- [ ] Tool execution telemetry
- [ ] Model usage metrics
- [ ] Debugging support

---

## Design Principles

### 1. Depend on abstractions

Components should depend on stable contracts such as:

```text
Agent
Model
Memory
Tool
```

rather than concrete implementations whenever practical.

---

### 2. Keep architectures isolated

Architecture-specific behavior belongs inside:

```text
agents/architectures/
```

For example:

```text
agents/architectures/react/
```

should contain ReAct execution logic.

MCP, model providers, and application code should not define the ReAct architecture.

---

### 3. Keep MCP independent from agent architectures

MCP provides capabilities.

It should not own:

- reasoning
- planning
- agent loops
- conversation management
- architecture selection

Agent architectures consume tools through the framework's tool abstraction.

---

### 4. Keep applications thin

Applications should expose or compose the framework rather than implement it.

The desired relationship is:

```text
CLI / API / React
       │
       ▼
   Application
       │
       ▼
      Agent
       │
   ┌───┼────┐
   ▼   ▼    ▼
 Model Memory Tools
```

The application should not directly implement agent orchestration.

---

### 5. Prefer explicit components

AgentForge intentionally avoids hiding the entire system behind a single framework abstraction.

Agents, architectures, models, memory, tools, and integrations remain visible components.

This makes the system easier to:

- understand
- test
- replace
- benchmark
- experiment with

---

### 6. Build only what experiments require

AgentForge is an experimentation framework.

New abstractions should be introduced when they solve a real architectural or experimental problem, not simply because the framework may need them in the future.

The objective is:

```text
Small Core
    ↓
Multiple Architectures
    ↓
Deterministic Tests
    ↓
Experiments
    ↓
Evidence
```

rather than building a large framework before there is something to evaluate.

---

## Current Direction

The immediate development sequence is:

```text
Clean Core
    │
    ▼
Direct Agent
    │
    ├──────────────► CLI
    │
    ├──────────────► FastAPI
    │                    │
    │                    ▼
    │                  React
    │
    ▼
ReAct Agent
    │
    ▼
Deterministic Tests
    │
    ▼
Architecture Evaluation
    │
    ▼
Additional Architectures
```

The long-term goal is to make different agent architectures comparable while keeping the underlying model, memory, tool, and infrastructure components independently replaceable.

---

## License

MIT
