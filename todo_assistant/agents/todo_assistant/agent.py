from typing import Optional

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from todo_assistant.agents.base import BaseAgent
from todo_assistant.agents.todo_assistant.output_parser import (
    TODOAssistantAgentOutput,
    TODOAssistantOutputParser,
)
from todo_assistant.prompts import TODO_ASSISTANT_AGENT_PROMPT

_TOOL_NAMES = ['todo_api_call', 'todo_query']


class TODOAssistantAgentInput(BaseModel):
    messages: list[BaseMessage]


class TODOAssistantAgent(BaseAgent[TODOAssistantAgentInput, TODOAssistantAgentOutput]):
    def __init__(self, agent: Runnable):
        self._agent = agent

    @classmethod
    def from_llm(cls, llm: ChatLiteLLM):
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                SystemMessagePromptTemplate(
                    prompt=PromptTemplate(input_variables=[], template=TODO_ASSISTANT_AGENT_PROMPT),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

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
                            {"enum": _TOOL_NAMES},
                        ],
                    },
                    "tool_input": {"title": "Tool input", "type": "string"},
                },
                "required": ["tool", "tool_input"],
            },
        }
        prompt = prompt.partial(tool_names='\n'.join(_TOOL_NAMES))
        agent = (
            prompt
            | llm.with_config({"run_name": "TODOAssistant_llm"}).bind(functions=[function_schema])
            | TODOAssistantOutputParser()
        )

        return cls(agent=agent)

    def invoke(
        self, input: TODOAssistantAgentInput, config: Optional[RunnableConfig] = None
    ) -> TODOAssistantAgentOutput:
        result_dict = self._agent.invoke(input={'messages': input.messages}, config=config)
        return TODOAssistantAgentOutput.parse_obj(result_dict)
