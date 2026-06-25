from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

app = FastAPI()

memory = MemorySaver()

client = None
tools = None
agent = None


@app.on_event("startup")
async def startup():
    global client, tools, agent

    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["mathserver.py"],
            "transport": "stdio",
        },
        "search": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    })

    tools = await client.get_tools()

    llm = ChatGroq(model="llama-3.1-70b-versatile")

    agent = create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
    )


class TaskRequest(BaseModel):
    user_id: str
    tasks: list[str]


@app.post("/daily-plan")
async def daily_plan(request: TaskRequest):

    prompt = f"""
You are a productivity assistant.

Return:

Priority Order:
Time Estimates:
Schedule:
Advice:

Tasks:
{request.tasks}
"""

    config = {
        "configurable": {
            "thread_id": f"daily-planner-{request.user_id}"
        }
    }

    inputs = {
        "messages": [{"role": "user", "content": prompt}]
    }

    final_event = None

    async for event in agent.astream(
        inputs,
        config=config,
        stream_mode="values"
    ):
        final_event = event

    return {
        "result": final_event["messages"][-1].content
    }