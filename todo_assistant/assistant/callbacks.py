import sys
from abc import ABC, abstractmethod

from todo_assistant.assistant.response import TODOAssistantResponse


class BaseAssistantResponseCallback(ABC):
    @abstractmethod
    def on_stream_new_token(self, token: str) -> None:
        pass

    @abstractmethod
    def on_stream_finish(self) -> None:
        pass

    @abstractmethod
    def on_response(self, final_response: TODOAssistantResponse) -> None:
        pass


class StdOutAssistantResponseCallback(BaseAssistantResponseCallback):
    def on_stream_new_token(self, token: str) -> None:
        sys.stdout.write(token)

    def on_stream_finish(self) -> None:
        sys.stdout.write('\n')

    def on_response(self, final_response: TODOAssistantResponse) -> None:
        pass
