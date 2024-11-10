import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
    CHAT_ID=os.getenv("CHAT_ID")