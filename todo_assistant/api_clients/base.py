from abc import ABC, abstractmethod

from todo_assistant.entities.task import CreateTaskRequest, Task


class BaseTaskAPIClient(ABC):
    @abstractmethod
    def get_by_id(self, id: str) -> Task:
        pass

    @abstractmethod
    def add(self, task_to_create: CreateTaskRequest) -> Task:
        pass

    @abstractmethod
    def update(self, task: Task) -> Task:
        pass

    @abstractmethod
    def delete(self, task_id: str) -> Task:
        pass
