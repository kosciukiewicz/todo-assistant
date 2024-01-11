import langchain_visualizer
from ice.recipe import FunctionBasedRecipe
from langchain.chat_models import ChatLiteLLM
from langchain.globals import set_debug, set_verbose
from langchain_openai import ChatOpenAI
from langchain_visualizer.hijacking import ice_hijack
from langchain_visualizer.llms.base import ChatLlmAsyncVisualizer, ChatLlmSyncVisualizer
from langgraph.pregel import Pregel

from todo_assistant.tools.api_calls.add import AddTaskTool
from todo_assistant.tools.api_calls.delete import DeleteTaskTool
from todo_assistant.tools.api_calls.update import UpdateTaskTool
from todo_assistant.tools.retrieval import TODORetrievalTool

set_debug(True)
set_verbose(True)
ice_hijack(ChatOpenAI, "_generate", ChatLlmSyncVisualizer)
ice_hijack(ChatOpenAI, "_agenerate", ChatLlmAsyncVisualizer)
ice_hijack(ChatLiteLLM, "_generate", ChatLlmSyncVisualizer)
ice_hijack(ChatLiteLLM, "_agenerate", ChatLlmAsyncVisualizer)
ice_hijack(TODORetrievalTool, "_run")
ice_hijack(AddTaskTool, "_run")
ice_hijack(DeleteTaskTool, "_run")
ice_hijack(UpdateTaskTool, "_run")
ice_hijack(Pregel, "invoke")


def visualize(function: FunctionBasedRecipe) -> None:
    langchain_visualizer.visualize(function)
