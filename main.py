import streamlit as st
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AgenticLanggraph MCP", layout="centered")
st.title("AgenticLanggraph: MCP ReAct Chatbot")
st.caption("Powered by LangGraph, ChatGroq, and FastMCP")

st.info("⚠️ Ensure the Weather server is running in the background: `python mcp-server/weatherserver.py`")

# Initialize app state only once per session
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.messages = []

# Configure the thread configuration for memory persistence
config = {"configurable": {"thread_id": "streamlit-session-1"}}

with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.memory = MemorySaver()
        st.session_state.messages = []
        st.rerun()

async def process_chat(prompt, message_placeholder):
    # Connect to MCP servers dynamically
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
    
    # Fetch tools asynchronously from the servers
    tools = await client.get_tools()
    from langchain_tavily import TavilySearch
    search_tool = TavilySearch(max_results=2)
    tools.append(search_tool)
    
    # Initialize the LLM with the user's requested model
    llm = ChatGroq(model="llama-3.1-8b-instant")
    
    # System prompt to prevent the model from hallucinating tool syntax in plain text
    system_prompt = (
        "You are a helpful assistant. Never output raw tool-calling syntax like <function=...> "
        "in your text responses. If asked about your tools, just describe them naturally in plain "
        "English without showing the exact syntax or XML tags."
    )
    
    # Create the ReAct agent with memory and the system prompt
    agent = create_react_agent(llm, tools, checkpointer=st.session_state.memory, prompt=system_prompt)
    
    inputs = {"messages": [HumanMessage(content=prompt)]}
    full_response = ""
    
    # Stream events to capture the final message
    async for event in agent.astream(inputs, config=config, stream_mode="values"):
        if "messages" in event:
            latest_msg = event["messages"][-1]
            if isinstance(latest_msg, AIMessage):
                # Ensure we only display the text content
                if latest_msg.content:
                    full_response = latest_msg.content
                    message_placeholder.markdown(full_response)
    
    return full_response

# Display previous messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
if prompt := st.chat_input("What is on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process response from the agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Invoke the agent asynchronously
        with st.spinner("Thinking & Using MCP Tools..."):
            try:
                full_response = asyncio.run(process_chat(prompt, message_placeholder))
                # Append the final assistant response to the session state
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error communicating with MCP Servers: {e}")

