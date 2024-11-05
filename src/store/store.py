from langchain_core.chat_history import BaseChatMessageHistory

from src.store.memory import InMemoryHistory

class MessagesStore:

    _store = {}

    def get_by_session_id(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryHistory()
        
        return self._store[session_id]
    
    @property
    def store(self):
        return self._store