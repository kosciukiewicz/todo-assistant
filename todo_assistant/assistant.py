from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.load import Serializable
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langsmith import traceable

from todo_assistant.prompts import TODO_ASSISTANT_INTRODUCTION_MESSAGE

_FINAL_MESSAGE = "Assistant decided to end conversation"
_STOP_INDICATOR = 'FINAL'
_TODO_ASSISTANT_LLM_NAME = "TODOAssistant_llm"
_TODO_ASSISTANT_AGENT_NAME = "TODOAssistant_agent"


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


class TODOAssistantResponse(Serializable):
    content: str
    is_final_response: bool

    @classmethod
    def create_final(cls, content: str = '') -> TODOAssistantResponse:
        return cls(
            content=content or _FINAL_MESSAGE,
            is_final_response=True,
        )


class TODOAssistant:
    def __init__(self, agent: Runnable, max_steps: int = 10) -> None:
        self._agent = agent.with_config({"run_name": _TODO_ASSISTANT_AGENT_NAME})
        self._conversation_history: list[BaseMessage] = []
        self._max_steps = max_steps

    @traceable(
        run_type="chain",
        name="Step",
    )
    def step(self) -> TODOAssistantResponse:
        response = self._agent.invoke({"messages": self._prepare_messages()})
        return self._handle_raw_response(response)

    @traceable(
        run_type="chain",
        name="Step",
    )
    async def astep(
        self, new_token_callback: BaseAssistantResponseCallback | None = None
    ) -> TODOAssistantResponse:
        if new_token_callback is None:
            new_token_callback = StdOutAssistantResponseCallback()

        async for event in self._agent.astream_events(
            {"messages": self._prepare_messages()},
            version="v1",
        ):
            name = event["name"]
            kind = event["event"]

            if kind == "on_chain_end" and name == _TODO_ASSISTANT_AGENT_NAME:
                if final_output := event['data']['output'].get('__end__'):
                    response = self._handle_raw_response(final_output)
                    new_token_callback.on_response(response)
                    return response
            if kind == "on_chat_model_stream" and name == _TODO_ASSISTANT_LLM_NAME:
                content = event["data"]["chunk"]
                if content and content != _STOP_INDICATOR:
                    new_token_callback.on_stream_new_token(content)
            if kind == "on_chat_model_end" and name == _TODO_ASSISTANT_LLM_NAME:
                new_token_callback.on_stream_finish()

        # No output, create final response
        response = TODOAssistantResponse.create_final()
        new_token_callback.on_response(response)
        return response

    def add_human_input(self, human_input: str) -> None:
        self._reset_state()
        self._conversation_history.append(HumanMessage(content=human_input))

    def _prepare_messages(self) -> list[BaseMessage]:
        return self._conversation_history or [
            # Single message is required, or astream_events logs broke because of jsonpatch
            # incompatibilities in log
            AIMessage(content=TODO_ASSISTANT_INTRODUCTION_MESSAGE)
        ]

    @staticmethod
    def _handle_raw_response(response: dict[str, Any]) -> TODOAssistantResponse:
        for message in reversed(response['messages']):
            if isinstance(message, AIMessage) and message.content:
                content = str(message.content)
                if _STOP_INDICATOR in content:
                    content = content.replace(_STOP_INDICATOR, '')
                    return TODOAssistantResponse.create_final(content.strip())
                else:
                    return TODOAssistantResponse(
                        content=content,
                        is_final_response=False,
                    )

        # No output, create final response
        return TODOAssistantResponse.create_final()

    def _reset_state(self) -> None:
        self._conversation_history = []
