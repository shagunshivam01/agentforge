from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPRegistry:
    def __init__(self):
        self.client = None
        self.tools = []

    async def initialize(self):
        """
        Connect to all MCP servers (currently only Tavily search).
        """

        self.client = MultiServerMCPClient({
            "search": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        })

        self.tools = await self.client.get_tools()

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

    def get_client(self):
        return self.client
