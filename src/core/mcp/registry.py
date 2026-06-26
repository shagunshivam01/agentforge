from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPRegistry:
    def __init__(self):
        self.client = None
        self.tools = []

    async def initialize(self):
        """
        Connect to all MCP servers.
        """

        server_config = {
            "search": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }

        self.client = MultiServerMCPClient(server_config)

        try:
            self.tools = await self.client.get_tools()
        except Exception as e:
            raise RuntimeError(f"MCP Registry init failed: {e}")

    def list_tools(self):
        """
        Return tool metadata for planner.
        """
        return [
            {
                "name": tool.name,
                "description": getattr(tool, "description", "")
            }
            for tool in self.tools
        ]

    def get_tools(self):
        """
        Return raw tool objects for execution layer.
        """
        return self.tools
    
    def get(self, name: str):
        for t in self.tools:
            if getattr(t, "name", None) == name:
                return t
        return None

    def get_client(self):
        return self.client

    def debug_tools(self):
        for t in self.tools:
            print(f"[TOOL] {t.name} -> {getattr(t, 'description', '')}")
    
    def tool_name(tool):
        return getattr(tool, "name", getattr(tool, "__name__", "unknown"))

    def tool_desc(tool):
        return getattr(tool, "description", "")

    def get_tool_schemas(self):
        """
        Structured schema for planner grounding.
        """

        schemas = []

        for tool in self.tools:
            schemas.append({
                "name": tool.name,
                "description": getattr(tool, "description", ""),
                "args": self._extract_schema(tool),
            })

        return schemas


    def _extract_schema(self, tool):
        """
        Best-effort schema extraction from MCP/LangChain tool.
        """

        schema = getattr(tool, "args_schema", None)

        if schema:
            try:
                return schema.model_json_schema()
            except Exception:
                return str(schema)

        return {}

