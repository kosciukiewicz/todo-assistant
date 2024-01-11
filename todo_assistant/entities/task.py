from __future__ import annotations

from enum import Enum
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel


class TaskStatus(Enum):
    NOT_STARTED = 'Not started'
    IN_PROGRESS = 'In progress'
    DONE = 'Done'


class TaskPriority(Enum):
    LOW = 'Low'
    MEDIUM = 'Medium'
    HIGH = 'High'


class CreateTaskRequest(BaseModel):
    title: str
    priority: TaskPriority
    work_estimation: int
    status: TaskStatus


class Task(BaseModel):
    id: str
    title: str
    priority: TaskPriority
    work_estimation: int
    status: TaskStatus

    @classmethod
    def from_document(cls, document: Document) -> Task:
        return cls(
            id=document.metadata['id'],
            title=document.metadata['name'],
            work_estimation=document.metadata['work estimation'],
            priority=TaskPriority(document.metadata['priority']),
            status=TaskStatus(document.metadata['status']),
        )

    def as_text(self) -> str:
        return " ".join(
            f"{param_name}=\"{param_value}\""
            for param_name, param_value in self.dict().items()
            if param_name != 'id'
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'name': self.title,
            'work_estimation': self.work_estimation,
            'status': self.status.value,
            'priority': self.priority.value,
        }
