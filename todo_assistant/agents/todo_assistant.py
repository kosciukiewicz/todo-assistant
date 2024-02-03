import json
from typing import Any, List, Optional, Type

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from todo_assistant.agents.base import BaseAgent
from todo_assistant.agents.todo_api import TODOAPIAgent
from todo_assistant.prompts import TODO_ASSISTANT_AGENT_PROMPT
from todo_assistant.tools.retrieval import TODORetrievalTool


class TODOAssistantAgentInput(BaseModel):
    messages: list[BaseMessage]


class TODOAssistantAgentOutput(BaseModel):
    next: str
    input: str | None = None


class _TODOAPIAgentToolInput(BaseModel):
    input: str = Field(description="Maximum one sentence, what you need to do with a tool.")


class _TODOAPIAgentTool(BaseTool):
    name = "todo_api_call"
    description = "Useful when you need to make operations on tasks"
    args_schema: Type[BaseModel] = _TODOAPIAgentToolInput
    agent: TODOAPIAgent

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()


class TODOAssistantAgent(BaseAgent[TODOAssistantAgentInput, TODOAssistantAgentOutput]):
    def __init__(self, agent: Runnable):
        self._agent = agent

    @classmethod
    def from_llm_and_agent_tools(
        cls,
        llm: ChatLiteLLM,
        todo_api_agent: TODOAPIAgent,
        retrieval_tool: TODORetrievalTool,
    ):
        todo_api_agent_tool = _TODOAPIAgentTool(agent=todo_api_agent)
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                SystemMessagePromptTemplate(
                    prompt=PromptTemplate(input_variables=[], template=TODO_ASSISTANT_AGENT_PROMPT),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        tool_names = [todo_api_agent_tool.name, retrieval_tool.name]
        function_schema = {
            "name": "tool_call",
            "description": "Select tool to call",
            "parameters": {
                "title": "Tool call",
                "type": "object",
                "properties": {
                    "tool": {
                        "title": "Tool",
                        "anyOf": [
                            {"enum": tool_names},
                        ],
                    },
                    "tool_input": {"title": "Tool input", "type": "string"},
                },
                "required": ["tool", "tool_input"],
            },
        }
        prompt = prompt.partial(tool_names='\n'.join(tool_names))
        agent = prompt | llm.bind(functions=[function_schema]) | TODOAssistantOutputParser()

        return cls(agent=agent)

    def invoke(
        self, input: TODOAssistantAgentInput, config: Optional[RunnableConfig] = None
    ) -> TODOAssistantAgentOutput:
        result_dict = self._agent.invoke(input={'messages': input.messages}, config=config)
        return TODOAssistantAgentOutput.parse_obj(result_dict)


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
        self, result: List[Generation], *, partial: bool = False
    ) -> TODOAssistantAgentOutput:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return self._parse_ai_message(message)

    def parse(self, text: str) -> TODOAssistantAgentOutput:
        raise ValueError("Can only parse messages")
