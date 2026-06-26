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
        Initialize MCP tool registry (connect all servers)
        """
        await self.registry.initialize()

    async def execute(self, user_id: str, input: str):
        """
        Main MCP execution loop:

        1. Save user message
        2. Build history
        3. Plan action
        4. Execute tool (if needed)
        5. Synthesize response
        6. Save assistant message
        """

        # MEMORY STORE
        self.memory.add(user_id, "user", input)

        # HISTORY
        history = self.memory.get(user_id)

        # TOOLS
        tool_schemas = self.registry.get_tool_schemas()

        # PLAN
        plan = await self.planner.plan(
            user_input=input,
            history=history,
            tool_schemas=tool_schemas,
        )

        print("=" * 50)
        print("PLAN")
        print(plan)
        print("=" * 50)

        # DIRECT RESPONSE PATH
        if not plan.requires_tools or len(plan.tool_calls) == 0:

            response = await self.planner.direct_response(
                user_input=input,
                history=history,
            )

            self.memory.add(user_id, "assistant", response)
            return response

        # EXECUTE TOOL CALLS
        tool_outputs = []

        for call in plan.tool_calls:

            tool = self.registry.get(call.name)

            if not tool:
                tool_outputs.append({
                    "error": f"Tool not found: {call.name}"
                })
                continue

            self.telemetry.log_step(user_id, {
                "tool": call.name,
                "input": call.arguments,
            })

            try:
                result = await tool.ainvoke(call.arguments)
            except Exception as e:
                result = {
                    "error": str(e),
                    "tool": call.name
                }

            tool_outputs.append(result)

        # SYNTHESIZE
        final_answer = await self.planner.synthesize(
            original_input=input,
            tool_outputs=[result],
            history=history,
        )

        # MEMORY STORE
        self.memory.add(user_id, "assistant", final_answer)

        return final_answer

