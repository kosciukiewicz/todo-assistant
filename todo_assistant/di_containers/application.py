from dependency_injector import containers, providers
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.document_loaders import NotionDBLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langgraph.pregel import Pregel

from todo_assistant.api_clients.notion import NotionDatabaseTaskAPIClient
from todo_assistant.assistant.assistant import TODOAssistant
from todo_assistant.di_containers.agents import (
    TODOAPIAgentContainer,
    TODOAPIAssistantAgentContainer,
    TODOAssistantAgentContainer,
)
from todo_assistant.di_containers.graph_builders import (
    TODOApiAgentGraphBuilder,
    TODOAssistantGraphBuilderContainer,
)
from todo_assistant.di_containers.tools import Tools
from todo_assistant.entities.task import Task
from todo_assistant.graphs.base import BaseGraphBuilder


def _init_vectorstore(notion_api_key: str, notion_database_id: str) -> VectorStore:
    loader = NotionDBLoader(
        integration_token=notion_api_key,
        database_id=notion_database_id,
    )
    docs = loader.load()

    texts = []
    metadatas = []
    for doc in docs:
        task = Task.from_document(doc)
        texts.append(task.as_text())
        metadatas.append(doc.metadata)

    if texts:
        return Chroma.from_texts(texts=texts, metadatas=metadatas, embedding=OpenAIEmbeddings())
    else:
        return Chroma(embedding_function=OpenAIEmbeddings())


class ApiClients(containers.DeclarativeContainer):
    config = providers.Configuration()

    task_api_client = providers.Singleton(
        NotionDatabaseTaskAPIClient,
        api_key=config.NOTION_API_KEY,
        database_id=config.NOTION_DATABASE_ID,
    )


class Loaders(containers.DeclarativeContainer):
    config = providers.Configuration()

    db_loader = providers.Singleton(
        NotionDBLoader,
        integration_token=config.NOTION_API_KEY,
        database_id=config.NOTION_DATABASE_ID,
    )


def _build_graph(graph_builder: BaseGraphBuilder) -> Pregel:
    return graph_builder.build_graph()


class Application(containers.DeclarativeContainer):
    config = providers.Configuration()

    api_clients = providers.Container(
        ApiClients,
        config=config,
    )

    llm = providers.Singleton(
        ChatLiteLLM,
        model=config.MODEL_NAME,
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
        streaming=True,
    )

    vectorstore = providers.Factory(
        _init_vectorstore,
        notion_api_key=config.NOTION_API_KEY,
        notion_database_id=config.NOTION_DATABASE_ID,
    )

    tools = providers.Container(
        Tools,
        config=config,
        llm=llm,
        task_api_client=api_clients.task_api_client,
        vectorstore=vectorstore,
    )

    todo_api_assistant_agent = providers.Container(
        TODOAPIAssistantAgentContainer,
        config=config,
        tools=tools,
        llm=llm,
    )

    todo_api_agent_graph_builder = providers.Container(
        TODOApiAgentGraphBuilder,
        config=config,
        tools=tools,
        todo_api_assistant_agent=todo_api_assistant_agent.agent,
    )

    todo_api_agent_graph = providers.Factory(
        _build_graph,
        graph_builder=todo_api_agent_graph_builder.graph_builder,
    )

    todo_api_agent = providers.Container(
        TODOAPIAgentContainer, config=config, llm=llm, todo_api_agent_graph=todo_api_agent_graph
    )

    todo_assistant_agent = providers.Container(
        TODOAssistantAgentContainer,
        config=config,
        llm=llm,
    )

    todo_assistant_graph_builder = providers.Container(
        TODOAssistantGraphBuilderContainer,
        config=config,
        tools=tools,
        todo_assistant_agent=todo_assistant_agent.agent,
        todo_api_agent=todo_api_agent.agent,
    )

    todo_assistant_graph = providers.Factory(
        _build_graph,
        graph_builder=todo_assistant_graph_builder.graph_builder,
    )

    todo_assistant = providers.Factory(TODOAssistant, agent=todo_assistant_graph)
