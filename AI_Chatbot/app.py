# the same code, just with a nice interface using the streamlit library
# run using: uv run streamlit run app.py

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

load_dotenv()

# page settings
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def get_agent():
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    tools = []

    return create_react_agent(model, tools)

agent = get_agent()

# maintaining chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 AI Assistant")
st.caption("Powered by Groq + LangGraph")

# to display the previous messages as the screen gets reloaded
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask me anything..."):

    # storing the user's message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        full_response = ""

        for chunk in agent.stream(
            {"messages": [HumanMessage(content=prompt)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:

                for message in chunk["agent"]["messages"]:

                    if isinstance(message, AIMessage):
                        full_response += message.content

                        response_placeholder.markdown(
                            full_response + "▌"
                        )

        response_placeholder.markdown(full_response)

    # storing bot's message
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )