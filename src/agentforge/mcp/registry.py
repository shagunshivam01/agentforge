from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPRegistry:
    def __init__(self, server_config: dict):
        self._client = MultiServerMCPClient(server_config)
        self._tools = []

    async def initialize(self) -> None:
        """
        Connect to MCP servers and load available tools.
        """

        try:
            self._tools = await self._client.get_tools()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MCP registry: {e}") from e

    def get_tools(self):
        """
        Return all registered MCP tools.
        """
        return self._tools
    
    def get(self, name: str):
        for t in self._tools:
            if getattr(t, "name", None) == name:
                return t
        return None

    def list_tools(self):
        """
        Return metadata for available MCP tools.
        """
        return [
            {
                "name": tool.name,
                "description": getattr(tool, "description", "")
            }
            for tool in self._tools
        ]
        
    def get_client(self):
        return self._client

    def debug_tools(self):
        for t in self._tools:
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

        for tool in self._tools:
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

