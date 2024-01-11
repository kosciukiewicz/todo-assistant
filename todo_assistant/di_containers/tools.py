from dependency_injector import containers, providers
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore

from todo_assistant.api_clients.base import BaseTaskAPIClient
from todo_assistant.tools.api_calls.add import AddTaskTool
from todo_assistant.tools.api_calls.delete import DeleteTaskTool
from todo_assistant.tools.api_calls.update import UpdateTaskTool
from todo_assistant.tools.retrieval import TODORetrievalTool
from todo_assistant.tools.search_task_id_by_name import SearchTaskIDByNameTool


def _get_api_call_tools(tools: list[providers.Factory[BaseTool]]) -> list[BaseTool]:
    return [tool() for tool in tools]


class Tools(containers.DeclarativeContainer):
    config = providers.Configuration()
    task_api_client: providers.Dependency[BaseTaskAPIClient] = providers.Dependency()
    vectorstore: providers.Dependency[VectorStore] = providers.Dependency()
    llm: providers.Dependency[ChatLiteLLM] = providers.Dependency()

    add_task_tool = providers.Factory(
        AddTaskTool,
        task_api_client=task_api_client,
        vectorstore=vectorstore,
    )

    update_task_tool = providers.Factory(
        UpdateTaskTool,
        task_api_client=task_api_client,
    )

    delete_task_tool = providers.Factory(
        DeleteTaskTool,
        task_api_client=task_api_client,
    )

    search_task_id_by_name_tool = providers.Factory(
        SearchTaskIDByNameTool,
        vectorstore=vectorstore,
    )

    api_call_tools = providers.Factory(
        _get_api_call_tools, tools=[add_task_tool, delete_task_tool, update_task_tool]
    )

    retrieval_tool = providers.Factory(
        TODORetrievalTool.from_llm_and_vectorstore,
        llm=llm,
        vectorstore=vectorstore,
    )
