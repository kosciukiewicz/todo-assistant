from abc import ABC
from typing import Generic, TypeVar

from langchain_core.runnables import Runnable

TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')


class BaseAgent(Runnable[TInput, TOutput], Generic[TInput, TOutput], ABC):
    pass
