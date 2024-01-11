import json
from typing import Any, List, Optional, Type

from langchain.chat_models import ChatLiteLLM
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
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

        llm_with_tools = llm.bind(
            functions=[
                format_tool_to_openai_function(t) for t in [todo_api_agent_tool, retrieval_tool]
            ]
        )

        agent = prompt | llm_with_tools | TODOAssistantOutputParser()

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

        if '<FINISH>' in message.content:
            return TODOAssistantAgentOutput(
                next="FINISH",
                input=None,
            )

        function_call = message.additional_kwargs.get("function_call", {})
        if function_call:
            function_name = function_call["name"]

            if len(function_call["arguments"].strip()) == 0:
                tool_input = {}
            else:
                tool_input = json.loads(function_call["arguments"], strict=False)

            return TODOAssistantAgentOutput(
                next=function_name,
                input=tool_input["input"],
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
