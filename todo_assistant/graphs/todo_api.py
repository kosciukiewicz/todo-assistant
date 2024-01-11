import operator
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel

from todo_assistant.agents.todo_api_assistant import (
    TODOAPIAssistantAgent,
    TODOAPIAssistantAgentInput,
    TODOAPIAssistantAgentOutput,
)
from todo_assistant.graphs.base.graph_builder import BaseGraphBuilder
from todo_assistant.graphs.base.nodes import BaseNode
from todo_assistant.tools.search_task_id_by_name import SearchTaskIDByNameTool


class TODOAPIGraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task_call: TODOAPIAssistantAgentOutput


class _TODOAPIAssistantNode(
    BaseNode[
        TODOAPIGraphState, dict[str, Any], TODOAPIAssistantAgentInput, TODOAPIAssistantAgentOutput
    ]
):
    def _build_input(self, state: TODOAPIGraphState) -> TODOAPIAssistantAgentInput:
        return TODOAPIAssistantAgentInput(messages=state['messages'])

    def _parse_output(
        self, state: TODOAPIGraphState, output: TODOAPIAssistantAgentOutput
    ) -> dict[str, Any]:
        return {
            'messages': [],
            'task_call': output,
        }


class _TODOAPIToolNode(BaseNode[TODOAPIGraphState, dict[str, Any], dict, str]):
    def _build_input(self, state: TODOAPIGraphState) -> dict:
        return state['task_call'].arguments

    def _parse_output(self, state: TODOAPIGraphState, output: str) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content=output,
                ),
            ]
        }


class TODOApiGraphBuilder(BaseGraphBuilder):
    def __init__(
        self,
        todo_api_assistant_agent: TODOAPIAssistantAgent,
        api_call_tools: list[BaseTool],
        search_task_id_by_name_tool: SearchTaskIDByNameTool,
    ):
        self._todo_api_assistant_agent = todo_api_assistant_agent
        self._api_call_tools = api_call_tools
        self._search_task_id_by_name_tool = search_task_id_by_name_tool

    def build_graph(self) -> Pregel:
        workflow = StateGraph(TODOAPIGraphState)

        workflow.add_node("TODOApiAgent", _TODOAPIAssistantNode(self._todo_api_assistant_agent))

        all_tools = self._api_call_tools + [self._search_task_id_by_name_tool]
        for tool in self._api_call_tools:
            todo_api_node = _TODOAPIToolNode(tool)
            workflow.add_node(tool.name, todo_api_node)
            workflow.add_edge(tool.name, END)

        search_task_by_name_node = _TODOAPIToolNode(self._search_task_id_by_name_tool)
        workflow.add_node(self._search_task_id_by_name_tool.name, search_task_by_name_node)
        workflow.add_edge(self._search_task_id_by_name_tool.name, 'TODOApiAgent')

        workflow.add_conditional_edges(
            "TODOApiAgent",
            self._route,
            {tool.name: tool.name for tool in all_tools} | {"end": END},
        )
        workflow.set_entry_point("TODOApiAgent")

        return workflow.compile()

    def _route(self, state: TODOAPIGraphState):
        # This is the router
        messages = state["messages"]
        last_message = messages[-1]

        if (
            state['task_call'].name == self._search_task_id_by_name_tool.name
            and last_message.content == "<NO TASK FOUND>"
        ):
            return "end"
        else:
            return state['task_call'].name
