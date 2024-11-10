from pinecone import Pinecone, ServerlessSpec, PineconeApiException
import streamlit as st

from langchain_ollama import OllamaEmbeddings
from langchain_core.exceptions import LangChainException

from src.config.config import Config

class VectorDatabase:

    def __init__(self, index_name: str, environment: str = "us-east-1") -> None:
        with st.status(label="DB Init", expanded=True) as status:
            try:
                st.write("Loading Config")
                self.config = Config()

                st.write("Connecting to Pinecone")
                self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
                self.index_name = index_name

                st.write("Setting Embedding")
                self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

                if index_name not in self.pc.list_indexes().names():
                    st.write("Creating Index")
                    self.pc.create_index(name=index_name, dimension=768, metric="cosine", spec=ServerlessSpec(cloud="aws", region=environment))

                self.index = self.pc.Index(name=index_name)

                status.update(label="DB Connected", state="complete")
            except LangChainException as e:
                status.update(label="Error: Couldn't Init DB", state="error")
            except PineconeApiException as e:
                status.update(label="Error: Couldn't Init DB", state="error")
            except Exception as e:
                status.update(label="Error: Couldn't Init DB", state="error")
    
    def add_message(self, message: str, role: str, conversation_id: str):
        """Adding an message to the vector database"""
        try:
            embedding = self.embeddings.embed_query(message)

            metadata = {"conversation_id": conversation_id, "message": message, "role": role}
            self.index.upsert(vectors=[
                {
                    "id": str(hash(message)),
                    "values": embedding,
                    "metadata": metadata
                }
            ], namespace=conversation_id, show_progress=True)

        except PineconeApiException as e:
            print(str(e))
            st.error("Error adding to DB", icon="💀")
        except Exception as e:
            print(str(e))
            st.error("Error adding to DB", icon="💀")

    def search_history(self, conversation_id: str, top_k = 5):
        query = { "conversation_id": { "$eq": conversation_id } }
        results = self.index.query(namespace="messages", filter=query, top_k=top_k, include_metadata=True)

        print("="*10, "Results", "="*10)
        print(results)

        return [result["metadata"]["message"] for result in results["matches"]]
    
    def clear_conversation(self, conversation_id: str):
        query = f"conversation_id:{conversation_id}"
        results = self.index.query(query, top_k=1000, include_metadata=True)
        ids_to_delete = [result["id"] for result in results["matches"]]

        if ids_to_delete:
            self.index.delete(ids=ids_to_delete)