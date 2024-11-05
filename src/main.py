import streamlit as st

from langchain_ollama import ChatOllama

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.store.store import MessagesStore

def main():
    st.set_page_config(page_title="Chatbot with Memory", page_icon="🤖")

    st.title("💬 Chatbot")
    st.caption("🚀 An Chatbot powered by LangChain and Qdrant")

    store = MessagesStore()

    history = store.get_by_session_id("1")

    history.add_ai_message(AIMessage(content="Hola"))

    if "history" not in st.session_state:
        st.session_state.history = history.messages if history else []

    for msg in st.session_state.history:
        if isinstance(msg, AIMessage):
            st.chat_message("ai").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

    base_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente que es bueno en {ability}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = base_prompt | ChatOllama(model="llama3.2:1b")

    chain_history = RunnableWithMessageHistory(
        chain,
        store.get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )

    if prompt := st.chat_input():
        st.session_state.history.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)

        response = chain_history.invoke(
            {"ability": "Asesoria Contable", "question": f"{prompt}"},
            config={"configurable": {"session_id": "foo"}}
        )

        print(response)

        ai_message = AIMessage(content=response.content)
        st.session_state.history.append(ai_message)

        st.chat_message("ai").write(response.content)

        # response = chat_handler.handle_query(prompt)
        # st.session_state.history.append({"role": "bot", "content": response})
        # st.chat_message("bot").write(response)