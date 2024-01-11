from typing import Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field


class SearchTaskIDByNameInput(BaseModel):
    task_name: str = Field(description="The name of the task")


class SearchTaskIDByNameTool(BaseTool):
    name = 'get_task_uuid'
    description = "Useful when you want to find uuid of specific task"
    args_schema: Type[BaseModel] = SearchTaskIDByNameInput
    vectorstore: VectorStore

    def _run(
        self, task_name: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> str | None:
        documents = self.vectorstore.search(
            query='', search_type='similarity', k=1, filter={'name': {'$eq': task_name}}
        )

        if documents:
            return f"Task id: \"{documents[0].metadata['id']}\""
        else:
            return "<NO TASK FOUND>"
