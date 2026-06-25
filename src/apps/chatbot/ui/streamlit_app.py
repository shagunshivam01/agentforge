import streamlit as st
import asyncio
import uuid

from apps.chatbot.app import ChatbotApp
from core.mcp.runtime import MCPRuntime

st.markdown(
    """
    <style>
    .stChatMessage {
        font-size: 17px;
        padding: 10px 14px;
    }

    .stChatMessage p {
        font-size: 17px;
    }

    /* user vs assistant differentiation */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"]) {
        border-radius: 10px;
    }

    .stChatInput textarea {
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="AgentForge MCP", layout="centered")
st.title("AgentForge MCP Chatbot")

# state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

@st.cache_resource
def load_app():
    runtime = MCPRuntime()
    asyncio.run(runtime.initialize())
    return ChatbotApp(runtime)

app = load_app()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        response = asyncio.run(
            app.run(
                user_id=st.session_state.thread_id,
                message=prompt
            )
        )

        placeholder.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
