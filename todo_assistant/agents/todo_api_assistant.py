import json
from typing import Any, List, Optional

from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from todo_assistant.agents.base import BaseAgent
from todo_assistant.prompts import TODO_API_ASSISTANT_AGENT_PROMPT
from todo_assistant.tools.search_task_id_by_name import SearchTaskIDByNameTool


class TODOAPIAssistantAgentOutput(BaseModel):
    name: str
    arguments: dict[str, Any]


class TODOAPIAssistantAgentInput(BaseModel):
    messages: list[BaseMessage]


class TODOAPIAssistantAgent(BaseAgent[TODOAPIAssistantAgentInput, TODOAPIAssistantAgentOutput]):
    def __init__(self, agent: Runnable):
        self._agent = agent

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: ChatLiteLLM,
        api_call_tools: list[BaseTool],
        search_task_id_by_name_tool: SearchTaskIDByNameTool,
    ):
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                SystemMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=[], template=TODO_API_ASSISTANT_AGENT_PROMPT
                    ),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        all_tools = api_call_tools + [search_task_id_by_name_tool]
        llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in all_tools])
        agent = prompt | llm_with_tools | APICallOutputParser()
        return cls(agent=agent)

    def invoke(
        self, input: TODOAPIAssistantAgentInput, config: Optional[RunnableConfig] = None
    ) -> TODOAPIAssistantAgentOutput:
        return self._agent.invoke(input={'messages': input.messages}, config=config)


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
        self, result: List[Generation], *, partial: bool = False
    ) -> TODOAPIAssistantAgentOutput:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return self._parse_ai_message(message)

    def parse(self, text: str) -> TODOAPIAssistantAgentOutput:
        raise ValueError("Can only parse messages")
