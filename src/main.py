import streamlit as st

# Importar el modelo ChatOllama y componentes de LangChain para el chatbot
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

# Importar módulos personalizados de configuración y base de datos
from src.store.memory import State
from src.config.config import Config
from src.db.vector import VectorDatabase

@st.cache_resource
def get_config():
    """Carga y retorna la configuración, almacenándola en caché para evitar duplicación de instancias."""
    config = Config()    
    return config

@st.cache_resource
def get_prompt_template():
    """Define el template de prompt para el chatbot, almacenado en caché para evitar recargas."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Eres un ayudante virtual" ),
            ("system", "Eres un Asesor Financiero que ayuda a calcular gastos y gestionar presupuestos"),
            ("system", "Responde de forma formal"),
            ("system", "Responde de forma precisa los calculos pedidos"),
            ("system", "Muestra el calculo realizado"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt

@st.cache_resource
def get_model():
    """Inicializa el modelo de Ollama y lo almacena en caché para evitar duplicación y errores."""
    return ChatOllama(model="llama3.2:1b")

def call_model(state: State):
    """Invoca el modelo con el estado actual de la conversación."""
    model = get_model()
    prompt = get_prompt_template()

    chain = prompt | model # Encadenar el prompt y el modelo para el flujo de conversación

    response = chain.invoke(state)

    return {"messages": response}

@st.cache_resource
def get_db():
    """Inicializa la base de datos vectorial para almacenar y buscar mensajes."""
    return VectorDatabase()

# Configurar el flujo de trabajo de estados del chatbot con un grafo de estados en caché
@st.cache_resource
def get_state_graph():
    """Configura el flujo de trabajo con un grafo de estados y un nodo de inicio."""
    workflow = StateGraph(state_schema=State)

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    return workflow

def main():
    """Función principal de la aplicación. Configura el chatbot y gestiona la conversación."""
    # Configuración de la página en Streamlit
    st.set_page_config(page_title="Asistente Financiero Virtual", page_icon="🪙")

    # Obtener configuración y definir ID de la conversación
    config = get_config()
    conversation_id="financial_assistant"

    # Conectar con la base de datos vectorial
    db = get_db()

    st.title("🏦 Asistente Financiero Virtual")
    st.caption("🚀 An Chatbot powered by LangChain")

    # Cargar historial de mensajes en la sesión si no están ya presentes
    if "messages" not in st.session_state:
        history = db.search_history(conversation_id=conversation_id)

        messages = [{ "role": history["metadatas"][index]["role"], "content": document } for index, document in enumerate(history["documents"])]

        st.session_state.messages = messages

    # Mostrar el historial de mensajes en el chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Obtener el flujo de trabajo de estados y compilar el grafo
    workflow = get_state_graph()
    app = workflow.compile(checkpointer=MemorySaver())  # Configurar guardado de memoria para seguimiento del estado

    # Definir configuración para el ID de la conversación
    config = {"configurable": {"thread_id": conversation_id}}

    # Procesar la entrada del usuario y actualizar la conversación
    if chat_input := st.chat_input("Como puedo ayudarte"):
        # Añadir mensaje del usuario al historial
        st.session_state.messages.append({ "role": "user", "content": chat_input })
        db.add_message(message=chat_input, role="user", conversation_id=conversation_id)

        # Mostrar el mensaje del usuario en el chat
        with st.chat_message("user"):
            st.markdown(chat_input)

        # Invocar el flujo de trabajo con los mensajes actuales y obtener respuesta del chatbot
        output = app.invoke({"messages": st.session_state.messages}, config)

        # Almacenar y mostrar la respuesta del chatbot
        response = output["messages"][-1].content
        db.add_message(message=response, role="ai", conversation_id=conversation_id)
        st.session_state.messages.append({ "role": "ai", "content": response })

        with st.chat_message("ai"):
            st.markdown(response)
    