import streamlit as st
import os

from langchain_ollama import ChatOllama

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph


from src.store.memory import State
from src.config.config import Config
from src.db.vector import VectorDatabase

@st.cache_resource
def get_config():
    config = Config()    
    return config

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
def get_db():
    print(1)
    return VectorDatabase(index_name=st.session_state.index_name)

def get_context(conversation_id: str):
    db = get_db()

    history = db.search_history(conversation_id=conversation_id)
    context = "\n".join(history)

    return context


@st.cache_resource
def get_state_graph():
    workflow = StateGraph(state_schema=State)

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    return workflow

def main():
    st.set_page_config(page_title="Chatbot with Memory", page_icon="🤖")

    config = get_config()

    conversation_id="chat_id"

    os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY

    with st.sidebar:
        st.title("💬 Chatbot")
        st.caption("🚀 An Chatbot powered by LangChain and Qdrant")

        language = st.selectbox("Cambiar idioma", ["Español", "Inglés", "Francés", "Alemán", "Italiano"])

    if "index_name" not in st.session_state:
        st.session_state.index_name = config.CHAT_ID

    db = get_db()

    if "messages" not in st.session_state:
        history = db.search_history(conversation_id=conversation_id)

        print("="*15, "History", "="*15)
        print(history)

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

    config = {"configurable": {"thread_id": conversation_id}}

    if chat_input := st.chat_input("Como puedo ayudarte"):
        input_messages = HumanMessage(chat_input)
        st.session_state.messages.append(input_messages)
        db.add_message(message=chat_input, role="user", conversation_id=conversation_id)

        with st.chat_message("user"):
            st.markdown(chat_input)

        output = app.invoke({"messages": st.session_state.messages, "language": language}, config)

        response = output["messages"][-1].content
        db.add_message(message=response, role="ai", conversation_id=conversation_id)
        st.session_state.messages.append(AIMessage(response))

        with st.chat_message("ai"):
            st.markdown(response)
    