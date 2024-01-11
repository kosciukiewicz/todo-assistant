from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from todo_assistant.agents.base import BaseAgent


class TODOAPIAgentInput(BaseModel):
    input: str


class TODOAPIAgentOutput(BaseModel):
    output: str


class TODOAPIAgent(BaseAgent[TODOAPIAgentInput, TODOAPIAgentOutput]):
    def __init__(self, agent: Runnable):
        self._agent = agent

    @classmethod
    def from_graph_builder(cls, todo_api_agent_graph: Runnable):
        agent = _enter_chain | todo_api_agent_graph
        return cls(agent=agent)

    def invoke(
        self, input: TODOAPIAgentInput, config: Optional[RunnableConfig] = None
    ) -> TODOAPIAgentOutput:
        results = self._agent.invoke(input=input.input, config=config)
        last_message = results['messages'][-1]
        return TODOAPIAgentOutput(output=last_message.content)


def _enter_chain(input_message: str):
    results = {
        "messages": [HumanMessage(content=input_message)],
    }
    return results
