from dependency_injector import containers, providers
from langchain_core.runnables import Runnable

from todo_assistant.graphs.base import BaseGraphBuilder
from todo_assistant.graphs.todo_api import TODOApiGraphBuilder
from todo_assistant.graphs.todo_assistant import TODOAssistantGraphBuilder


class GraphBuilderContainer(containers.DeclarativeContainer):
    graph_builder = providers.Factory[BaseGraphBuilder]


class TODOApiAgentGraphBuilder(GraphBuilderContainer):
    config = providers.Configuration()
    tools = providers.DependenciesContainer()
    todo_api_assistant_agent = providers.Dependency(
        instance_of=Runnable  # type: ignore[type-abstract]
    )

    graph_builder: providers.Factory[TODOApiGraphBuilder] = providers.Factory(
        TODOApiGraphBuilder,
        todo_api_assistant_agent=todo_api_assistant_agent,
        api_call_tools=tools.api_call_tools,
        search_task_id_by_name_tool=tools.search_task_id_by_name_tool,
    )


class TODOAssistantGraphBuilderContainer(GraphBuilderContainer):
    config = providers.Configuration()
    tools = providers.DependenciesContainer()
    todo_assistant_agent = providers.Dependency(instance_of=Runnable)  # type: ignore[type-abstract]
    todo_api_agent = providers.Dependency(instance_of=Runnable)  # type: ignore[type-abstract]

    graph_builder = providers.Factory(
        TODOAssistantGraphBuilder,
        todo_assistant_agent=todo_assistant_agent,
        todo_api_agent=todo_api_agent,
        retrieval_tool=tools.retrieval_tool,
    )
