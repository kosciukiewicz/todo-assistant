from __future__ import annotations

from langchain_core.load import Serializable

_FINAL_MESSAGE = "Assistant decided to end conversation"


class TODOAssistantResponse(Serializable):
    content: str
    is_final_response: bool

    @classmethod
    def create_final(cls, content: str = '') -> TODOAssistantResponse:
        return cls(
            content=content or _FINAL_MESSAGE,
            is_final_response=True,
        )
