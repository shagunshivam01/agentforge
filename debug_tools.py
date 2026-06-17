import asyncio
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
import os

load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mcp-server/mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    search_tool = TavilySearch(max_results=2)
    tools.append(search_tool)
    
    llm = ChatGroq(model="llama-3.1-8b-instant")
    agent = create_react_agent(llm, tools)
    
    response = await agent.ainvoke({"messages": [HumanMessage(content="hello")]})
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
