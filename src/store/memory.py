from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from pydantic import BaseModel, Field

class InMemoryHistory(BaseChatMessageHistory, BaseModel):

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []