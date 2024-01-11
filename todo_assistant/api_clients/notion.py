import typing
from typing import Any

from notion_client import Client

from todo_assistant.api_clients.base import BaseTaskAPIClient
from todo_assistant.entities.task import CreateTaskRequest, Task, TaskPriority, TaskStatus


class NotionDatabaseTaskAPIClient(BaseTaskAPIClient):
    def __init__(
        self,
        api_key: str,
        database_id: str,
    ):
        self._client = Client(auth=api_key)
        self._database_id = database_id

    def get_by_id(self, id: str) -> Task:
        response = self._client.pages.retrieve(page_id=id)
        return self._create_task_from_response(response=typing.cast(dict[str, Any], response))

    def add(self, task_to_create: CreateTaskRequest) -> Task:
        response = self._client.pages.create(
            parent={
                'database_id': self._database_id,
            },
            properties=self._create_request_properties_from_task_properties(
                title=task_to_create.title,
                priority=task_to_create.priority,
                status=task_to_create.status,
                work_estimation=task_to_create.work_estimation,
            ),
        )
        return self._create_task_from_response(response=typing.cast(dict[str, Any], response))

    def update(self, task: Task) -> Task:
        response = self._client.pages.update(
            page_id=task.id,
            properties=self._create_request_properties_from_task_properties(
                title=task.title,
                priority=task.priority,
                status=task.status,
                work_estimation=task.work_estimation,
            ),
        )
        return self._create_task_from_response(response=typing.cast(dict[str, Any], response))

    def delete(self, task_id: str) -> Task:
        response = self._client.pages.update(
            page_id=task_id,
            archived=True,
        )
        return self._create_task_from_response(response=typing.cast(dict[str, Any], response))

    @staticmethod
    def _create_request_properties_from_task_properties(
        title: str,
        work_estimation: int,
        priority: TaskPriority,
        status: TaskStatus,
    ) -> dict[str, Any]:
        return {
            "Name": {"title": [{"text": {"content": title}}]},
            "Work estimation": {"number": work_estimation},
            "Priority": {"select": {"name": priority.value}},
            "Status": {"status": {"name": status.value}},
        }

    @staticmethod
    def _create_task_from_response(response: dict[str, Any]) -> Task:
        return Task(
            id=response['id'],
            title=response['properties']['Name']['title'][0]['plain_text'],
            priority=TaskPriority(response['properties']['Priority']['select']['name']),
            work_estimation=response['properties']['Work estimation']['number'],
            status=TaskStatus(response['properties']['Status']['status']['name']),
        )
