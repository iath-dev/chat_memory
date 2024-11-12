import streamlit as st

from langchain_ollama import OllamaEmbeddings
from langchain_core.exceptions import LangChainException

from langchain_chroma import Chroma

class VectorDatabase:
    """Base de datos  vectorial"""
    def __init__(self) -> None:
        """
        Inicializacion de la base de datos vectorial
        """
        try:
            # Iniciando el EMbdedding necesario para ChromaDB
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            # Creacion de la instancia de ChromaDB
            self.db = Chroma(collection_name="chat_collection", embedding_function=self.embeddings, persist_directory="./chroma_db")

            print("DB Init")
        except LangChainException as e:
            print("Error: Couldn't Init DB")
        except Exception as e:
            print("Error: Couldn't Init DB")
    
    def add_message(self, message: str, role: str, conversation_id: str):
        """Agregando un nuevo mensaje a la base de datos"""
        try:
            # Agregando mensaje a la base de datos de ChromaDB
            self.db.add_texts(texts=[message], metadatas=[{ "role": role, "conversation_id": conversation_id }])

        except PineconeApiException as e:
            print(str(e))
            st.error("Error adding to DB", icon="💀")
        except Exception as e:
            print(str(e))
            st.error("Error adding to DB", icon="💀")

    def get_all(self):
        """Obtener todos los registros de la base de datos"""
        results = self.db.get()

        return results

    def search_history(self, conversation_id: str):
        """Busqueda de los mensajes ne la base de datos por conversacion"""
        results = self.db.get(where={"conversation_id": conversation_id})

        return results