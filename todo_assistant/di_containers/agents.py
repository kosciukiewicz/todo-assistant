from dependency_injector import containers, providers
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.runnables import Runnable
from langgraph.pregel import Pregel

from todo_assistant.agents.todo_api import TODOAPIAgent
from todo_assistant.agents.todo_api_assistant import TODOAPIAssistantAgent
from todo_assistant.agents.todo_assistant import TODOAssistantAgent
from todo_assistant.tools.retrieval import TODORetrievalTool


class AgentContainer(containers.DeclarativeContainer):
    agent = providers.Factory[Runnable]


class TODOAPIAssistantAgentContainer(AgentContainer):
    config = providers.Configuration()
    tools = providers.DependenciesContainer()
    llm: providers.Dependency[ChatLiteLLM] = providers.Dependency()

    agent = providers.Factory(
        TODOAPIAssistantAgent.from_llm_and_tools,
        llm=llm,
        api_call_tools=tools.api_call_tools,
        search_task_id_by_name_tool=tools.search_task_id_by_name_tool,
    )


class TODOAssistantAgentContainer(AgentContainer):
    config = providers.Configuration()
    llm: providers.Dependency[ChatLiteLLM] = providers.Dependency()
    todo_api_agent = providers.Dependency(instance_of=TODOAPIAgent)
    retrieval_tool = providers.Dependency(instance_of=TODORetrievalTool)

    agent = providers.Factory(
        TODOAssistantAgent.from_llm_and_agent_tools,
        llm=llm,
        todo_api_agent=todo_api_agent,
        retrieval_tool=retrieval_tool,
    )


class TODOAPIAgentContainer(AgentContainer):
    config = providers.Configuration()
    llm: providers.Dependency[ChatLiteLLM] = providers.Dependency()
    todo_api_agent_graph = providers.Dependency(instance_of=Pregel)

    agent = providers.Factory(
        TODOAPIAgent.from_graph_builder, todo_api_agent_graph=todo_api_agent_graph
    )
