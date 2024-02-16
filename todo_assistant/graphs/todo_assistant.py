import json
import operator
from abc import ABC, abstractmethod
from typing import Annotated, Any, Generic, Sequence, TypedDict, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel

from todo_assistant.agents.base import BaseAgent
from todo_assistant.agents.todo_api import TODOAPIAgentInput, TODOAPIAgentOutput
from todo_assistant.agents.todo_assistant import (
    TODOAssistantAgent,
    TODOAssistantAgentInput,
    TODOAssistantAgentOutput,
)
from todo_assistant.graphs.base import BaseGraphBuilder, BaseNode
from todo_assistant.tools.retrieval import TODORetrievalTool


class TODOAssistantGraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    input: str | None


TRunnableInput = TypeVar('TRunnableInput')
TRunnableOutput = TypeVar('TRunnableOutput')


class _BaseTODOAssistantToolNode(
    BaseNode[TODOAssistantGraphState, dict[str, Any], TRunnableInput, TRunnableOutput],
    Generic[TRunnableInput, TRunnableOutput],
    ABC,
):
    @abstractmethod
    def _parse_content(self, output: TRunnableOutput) -> str:
        pass

    def _parse_output(
        self, state: TODOAssistantGraphState, output: TRunnableOutput
    ) -> dict[str, Any]:
        content = self._parse_content(output)
        return {
            'messages': [
                AIMessage(
                    content='',
                    additional_kwargs={
                        'function_call': {
                            'arguments': json.dumps(
                                {'tool_input': state['input'], 'tool': state['next']}
                            ),
                            'name': 'tool_call',
                        }
                    },
                ),
                FunctionMessage(content=content, name=state['next']),
            ]
        }


class _TODOAssistantNode(
    BaseNode[
        TODOAssistantGraphState,
        TODOAssistantGraphState,
        TODOAssistantAgentInput,
        TODOAssistantAgentOutput,
    ]
):
    def _build_input(self, state: TODOAssistantGraphState) -> TODOAssistantAgentInput:
        return TODOAssistantAgentInput(messages=state['messages'])

    def _parse_output(
        self, state: TODOAssistantGraphState, output: TODOAssistantAgentOutput
    ) -> TODOAssistantGraphState:
        messages = []
        if output.next == 'RESPOND':
            messages.append(AIMessage(content=output.input))

        return {
            'messages': messages,
            'next': output.next,
            'input': output.input,
        }


class _TODOAPIAgentNode(
    _BaseTODOAssistantToolNode[TODOAPIAgentInput, TODOAPIAgentOutput],
):
    def _build_input(self, state: TODOAssistantGraphState) -> TODOAPIAgentInput:
        return TODOAPIAgentInput(input=state['input'])

    def _parse_content(self, output: TODOAPIAgentOutput) -> str:
        return output.output


class _TODORetrievalToolNode(
    _BaseTODOAssistantToolNode[str, str],
):
    def _build_input(self, state: TODOAssistantGraphState) -> str:
        if retrieval_input := state['input']:
            return retrieval_input
        else:
            raise ValueError("Cannot run retrieval with empty query")

    def _parse_content(self, output: str) -> str:
        return output


class TODOAssistantGraphBuilder(BaseGraphBuilder):
    def __init__(
        self,
        todo_assistant_agent: TODOAssistantAgent,
        todo_api_agent: BaseAgent[TODOAPIAgentInput, TODOAPIAgentOutput],
        retrieval_tool: TODORetrievalTool,
    ):
        self._todo_assistant_agent = todo_assistant_agent
        self._todo_api_agent = todo_api_agent
        self._retrieval_tool = retrieval_tool

    def build_graph(self) -> Pregel:
        workflow = StateGraph(TODOAssistantGraphState)

        workflow.add_node(
            "TodoAssistant",
            _TODOAssistantNode(self._todo_assistant_agent),
        )

        workflow.add_node("APIAgent", _TODOAPIAgentNode(self._todo_api_agent))
        workflow.add_node("RetrievalTool", _TODORetrievalToolNode(self._retrieval_tool))

        workflow.add_conditional_edges(
            "TodoAssistant",
            lambda state: state["next"],
            {
                "todo_api_call": "APIAgent",
                "todo_query": "RetrievalTool",
                "RESPOND": END,
            },
        )
        workflow.add_edge("APIAgent", 'TodoAssistant')
        workflow.add_edge("RetrievalTool", 'TodoAssistant')

        workflow.set_entry_point("TodoAssistant")

        return workflow.compile()
