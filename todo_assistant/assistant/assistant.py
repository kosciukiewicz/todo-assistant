from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langsmith import traceable

from todo_assistant.assistant.callbacks import (
    BaseAssistantResponseCallback,
    StdOutAssistantResponseCallback,
)
from todo_assistant.assistant.response import TODOAssistantResponse
from todo_assistant.prompts import STOP_INDICATOR, TODO_ASSISTANT_INTRODUCTION_MESSAGE

_TODO_ASSISTANT_LLM_NAME = "TODOAssistant_llm"
_TODO_ASSISTANT_AGENT_NAME = "TODOAssistant_agent"
_TODO_ASSISTANT_NAME = "TODOAssistant"


class TODOAssistant:
    def __init__(self, agent: Runnable, max_steps: int = 10, session_id: str | None = None) -> None:
        agent_with_history = (
            self._insert_human_message
            | agent.with_config({"run_name": _TODO_ASSISTANT_AGENT_NAME})
            | self._filter_response_messages
        )
        self._session_id = session_id or uuid4().hex
        self._agent = RunnableWithMessageHistory(
            agent_with_history.with_config(  # type: ignore[arg-type]
                {"run_name": _TODO_ASSISTANT_NAME}
            ),
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="messages",
            output_messages_key="messages",
        ).with_config({"configurable": {"session_id": self._session_id}})
        self._max_steps = max_steps
        self._conversation_history_store: dict[str, ChatMessageHistory] = {}

    @traceable(
        run_type="chain",
        name="Step",
    )
    def init(self) -> TODOAssistantResponse:
        return self.step(human_input=TODO_ASSISTANT_INTRODUCTION_MESSAGE)

    @traceable(
        run_type="chain",
        name="Step",
    )
    def step(self, human_input: str) -> TODOAssistantResponse:
        response = self._agent.invoke({"input": human_input})
        return self._handle_raw_response(response)

    @traceable(
        run_type="chain",
        name="Step",
    )
    async def ainit(
        self, new_token_callback: BaseAssistantResponseCallback | None = None
    ) -> TODOAssistantResponse:
        return await self.astep(
            human_input=TODO_ASSISTANT_INTRODUCTION_MESSAGE, new_token_callback=new_token_callback
        )

    @traceable(
        run_type="chain",
        name="Step",
    )
    async def astep(
        self, human_input: str, new_token_callback: BaseAssistantResponseCallback | None = None
    ) -> TODOAssistantResponse:
        if new_token_callback is None:
            new_token_callback = StdOutAssistantResponseCallback()

        async for event in self._agent.astream_events(
            {"input": human_input},
            version="v1",
        ):
            name = event["name"]
            kind = event["event"]

            if kind == "on_chain_end" and name == _TODO_ASSISTANT_NAME:
                if final_output := event['data']['output']:
                    response = self._handle_raw_response(final_output)
                    new_token_callback.on_response(response)
                    return response
            if kind == "on_chat_model_stream" and name == _TODO_ASSISTANT_LLM_NAME:
                content = event["data"]["chunk"]
                if content and content != STOP_INDICATOR:
                    new_token_callback.on_stream_new_token(content)
            if kind == "on_chat_model_end" and name == _TODO_ASSISTANT_LLM_NAME:
                new_token_callback.on_stream_finish()

        # No output, create final response
        response = TODOAssistantResponse.create_final()
        new_token_callback.on_response(response)
        return response

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._conversation_history_store:
            self._conversation_history_store[session_id] = ChatMessageHistory()
        return self._conversation_history_store[session_id]

    @staticmethod
    def _insert_human_message(input_message: dict[str, Any]) -> dict[str, Any]:
        results = {
            "messages": [*input_message['messages'], HumanMessage(content=input_message['input'])],
        }
        return results

    @staticmethod
    def _filter_response_messages(
        result: dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        if 'messages' in result:
            messages = result['messages']
        else:
            messages = result['__end__']['messages']

        filtered_messages = []

        for message in messages:
            if isinstance(message, AIMessage | HumanMessage) and message.content:
                filtered_messages.append(message)

        if config:
            message_history_len = 1 + len(config['configurable']['message_history'].messages)
        else:
            message_history_len = 0

        messages = filtered_messages[message_history_len:]
        return {'messages': messages}

    @staticmethod
    def _handle_raw_response(response: dict[str, Any]) -> TODOAssistantResponse:
        if len(response['messages']) == 1:
            last_message = response['messages'][-1]
            if isinstance(last_message, AIMessage) and last_message.content:
                content = str(last_message.content)
                if STOP_INDICATOR in content:
                    content = content.replace(STOP_INDICATOR, '')
                    return TODOAssistantResponse.create_final(content.strip())
                else:
                    return TODOAssistantResponse(
                        content=content,
                        is_final_response=False,
                    )

            raise ValueError("No message from TODOAssistant found")
        else:
            raise ValueError("Expected exactly one message from TODOAssistant")
