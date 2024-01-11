from typing import Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from pydantic.fields import Field
from pydantic.main import BaseModel

from todo_assistant.api_clients.base import BaseTaskAPIClient
from todo_assistant.entities.task import CreateTaskRequest, TaskPriority, TaskStatus


class AddTaskInput(BaseModel):
    task_name: str = Field(description="The name of the task")
    task_params: str = Field(
        description="Required json object containing additional properties for the task, if there"
        " are none pass empty dict."
    )


class AddTaskTool(BaseTool):
    name = "add_task"
    description = "Useful when you want to add new task to tasks board"
    args_schema: Type[BaseModel] = AddTaskInput
    task_api_client: BaseTaskAPIClient
    vectorstore: VectorStore

    def _run(
        self, task_name: str, task_params: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        task = self.task_api_client.add(
            task_to_create=CreateTaskRequest(
                title=task_name,
                priority=TaskPriority.HIGH,
                work_estimation=2,
                status=TaskStatus.IN_PROGRESS,
            )
        )
        self.vectorstore.add_texts(texts=[task.as_text()], metadatas=[task.as_metadata()])
        return f"Added \"{task_name}\" task to board with id=\"{task.id}\""
