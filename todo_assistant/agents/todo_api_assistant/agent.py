from typing import Optional

from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import BaseMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel

from todo_assistant.agents.base import BaseAgent
from todo_assistant.agents.todo_api_assistant.output_parser import (
    APICallOutputParser,
    TODOAPIAssistantAgentOutput,
)
from todo_assistant.prompts import TODO_API_ASSISTANT_AGENT_PROMPT
from todo_assistant.tools.search_task_id_by_name import SearchTaskIDByNameTool


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
        llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in all_tools])
        agent = prompt | llm_with_tools | APICallOutputParser()
        return cls(agent=agent)

    def invoke(
        self, input: TODOAPIAssistantAgentInput, config: Optional[RunnableConfig] = None
    ) -> TODOAPIAssistantAgentOutput:
        return self._agent.invoke(input={'messages': input.messages}, config=config)
