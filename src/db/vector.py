from pinecone import Pinecone, ServerlessSpec, PineconeApiException
import streamlit as st

from langchain_ollama import OllamaEmbeddings
from langchain_core.exceptions import LangChainException

from langchain_chroma import Chroma

from src.config.config import Config

class VectorDatabase:

    def __init__(self) -> None:
        try:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

            self.db = Chroma(collection_name="chat_collection", embedding_function=self.embeddings, persist_directory="./chroma_db")

            print("DB Init")
        except LangChainException as e:
            print("Error: Couldn't Init DB")
        except PineconeApiException as e:
            print("Error: Couldn't Init DB")
        except Exception as e:
            print("Error: Couldn't Init DB")
    
    def add_message(self, message: str, role: str, conversation_id: str):
        """Adding an message to the vector database"""
        try:
            self.db.add_texts(texts=[message], metadatas=[{ "role": role, "conversation_id": conversation_id }])

        except PineconeApiException as e:
            print(str(e))
            st.error("Error adding to DB", icon="💀")
        except Exception as e:
            print(str(e))
            st.error("Error adding to DB", icon="💀")

    def get_all(self):
        results = self.db.get()

        return results

    def search_history(self, conversation_id: str):
        results = self.db.get(where={"conversation_id": conversation_id})

        return results