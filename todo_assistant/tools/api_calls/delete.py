from typing import Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from todo_assistant.api_clients.base import BaseTaskAPIClient


class DeleteTaskInput(BaseModel):
    task_id: str = Field(description="Uuid of the task")


class DeleteTaskTool(BaseTool):
    name = "delete_task"
    description = "useful when you want to delete task from tasks board"
    task_api_client: BaseTaskAPIClient
    args_schema: Type[BaseModel] = DeleteTaskInput

    def _run(self, task_id: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        self.task_api_client.delete(task_id)
        return f"Removed task with id=\"{task_id}\""
