from core.mcp.planner import MCPPlanner
from core.mcp.registry import MCPRegistry
from core.mcp.telemetry import MCPTelemetry
from core.memory.conversation import ConversationMemory


class MCPRuntime:
    def __init__(self):
        self.registry = MCPRegistry()
        self.planner = MCPPlanner()
        self.telemetry = MCPTelemetry()
        self.memory = ConversationMemory()

    async def initialize(self):
        """
        Connect to MCP servers
        """
        await self.registry.initialize()

    async def execute(self, user_id: str, input: str):
        """
        Main execution loop (no LangGraph, pure MCP runtime)
        """
        
        self.memory.add(
            user_id,
            "user",
            input,
        )

        history = self.memory.get(user_id)
        tools = self.registry.get_tools()

        # PLAN (decide which tool to use)
        plan = await self.planner.plan(
            user_input=input,
            history=history,
            tools=tools,
        )

        if plan["tool"] in [None, "", "none"]:
            response = await self.planner.direct_response(input, history)

            self.memory.add(
                user_id,
                "assistant",
                response,
            )

            return response

        tool_name = plan["tool"]
        tool_input = plan["input"]

        tool = next(
            (
                t
                for t in self.registry.get_tools()
                if t.name == tool_name
            ),
            None,
        )

        if not tool:
            raise Exception(f"Tool not found: {tool_name}")

        result = await tool.ainvoke(tool_input)

        results = [result]

        # SYNTHESIZE FINAL RESPONSE
        final_answer = await self.planner.synthesize(
            original_input=input,
            tool_outputs=results
        )

        # Save context in memory
        self.memory.add(
            user_id,
            "assistant",
            final_answer,
        )

        return final_answer
