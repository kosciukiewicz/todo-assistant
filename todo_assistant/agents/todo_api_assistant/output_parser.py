import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from pydantic import BaseModel


class TODOAPIAssistantAgentOutput(BaseModel):
    name: str
    arguments: dict[str, Any]


class APICallOutputParser(BaseOutputParser[TODOAPIAssistantAgentOutput]):
    @property
    def _type(self) -> str:
        return "api-call-output-parser"

    @staticmethod
    def _parse_ai_message(message: BaseMessage) -> TODOAPIAssistantAgentOutput:
        """Parse an AI message."""
        if not isinstance(message, AIMessage):
            raise TypeError(f"Expected an AI message got {type(message)}")

        function_call = message.additional_kwargs["function_call"]
        function_name = function_call["name"]

        if len(function_call["arguments"].strip()) == 0:
            tool_input = {}
        else:
            tool_input = json.loads(function_call["arguments"], strict=False)

        return TODOAPIAssistantAgentOutput(
            name=function_name,
            arguments=tool_input,
        )

    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> TODOAPIAssistantAgentOutput:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return self._parse_ai_message(message)

    def parse(self, text: str) -> TODOAPIAssistantAgentOutput:
        raise ValueError("Can only parse messages")
