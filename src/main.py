import streamlit as st

from langchain_ollama import ChatOllama

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from src.store.memory import State

@st.cache_resource
def get_prompt_template():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Eres un ayudante virtual" ),
            ("system", "Response como un mayordomo"), 
            ("system", "Debes responder en {language}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt

@st.cache_resource
def get_model():
    return ChatOllama(model="llama3.2:1b")

def call_model(state: State):
    model = get_model()
    prompt = get_prompt_template()

    chain = prompt | model

    response = chain.invoke(state)

    return {"messages": response}

@st.cache_resource
def get_state_graph():
    workflow = StateGraph(state_schema=State)

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    return workflow

def main():
    st.set_page_config(page_title="Chatbot with Memory", page_icon="🤖")

    with st.sidebar:
        st.title("💬 Chatbot")
        st.caption("🚀 An Chatbot powered by LangChain and Qdrant")

        language = st.selectbox("Cambiar idioma", ["Español", "Inglés", "Francés", "Alemán", "Italiano"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        if isinstance(msg, AIMessage):
            with st.chat_message("ai"):
                st.markdown(msg.content)

    workflow = get_state_graph()

    app = workflow.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "abc123"}}

    if chat_input := st.chat_input("Como puedo ayudarte"):
        input_messages = HumanMessage(chat_input)
        st.session_state.messages.append(input_messages)

        with st.chat_message("user"):
            st.markdown(chat_input)

        output = app.invoke({"messages": st.session_state.messages, "language": language}, config)

        response = output["messages"][-1].content
        st.session_state.messages.append(AIMessage(response))

        with st.chat_message("ai"):
            st.markdown(response)
    