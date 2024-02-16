import json

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from pydantic import BaseModel


class TODOAssistantAgentOutput(BaseModel):
    next: str
    input: str | None = None


class TODOAssistantOutputParser(BaseOutputParser[TODOAssistantAgentOutput]):
    @property
    def _type(self) -> str:
        return "todo-assistant-output-parser"

    @staticmethod
    def _parse_ai_message(message: BaseMessage) -> TODOAssistantAgentOutput:
        """Parse an AI message."""
        if not isinstance(message, AIMessage):
            raise TypeError(f"Expected an AI message got {type(message)}")

        function_call = message.additional_kwargs.get("function_call", {})
        if function_call:
            arguments = json.loads(function_call["arguments"], strict=False)

            return TODOAssistantAgentOutput(
                next=arguments["tool"],
                input=arguments["tool_input"],
            )
        else:
            return TODOAssistantAgentOutput(
                next="RESPOND",
                input=message.content,
            )

    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> TODOAssistantAgentOutput:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return self._parse_ai_message(message)

    def parse(self, text: str) -> TODOAssistantAgentOutput:
        raise ValueError("Can only parse messages")
