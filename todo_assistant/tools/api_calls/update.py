import json
from typing import Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic.fields import Field
from pydantic.main import BaseModel

from todo_assistant.api_clients.base import BaseTaskAPIClient
from todo_assistant.entities.task import TaskPriority, TaskStatus


class UpdateTaskInput(BaseModel):
    task_id: str = Field(description="Uuid of the task")
    task_params: str = Field(
        description="Required json object containing additional properties to update for the task"
    )


class UpdateTaskTool(BaseTool):
    name = "update_task"
    description = "useful when you want to update some properties of the task in tasks board"
    args_schema: Type[BaseModel] = UpdateTaskInput
    task_api_client: BaseTaskAPIClient

    def _run(
        self,
        task_id: str,
        task_params: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        task = self.task_api_client.get_by_id(task_id)
        for param_name, param_value in json.loads(task_params).items():
            match param_name:
                case 'status':
                    task.status = TaskStatus(str(param_value).capitalize())
                case 'priority':
                    task.priority = TaskPriority(str(param_value).capitalize())
                case _:
                    setattr(task, param_name, param_value)

        task = self.task_api_client.update(task)
        return f"Updated task with id=\"{task.id}\""
