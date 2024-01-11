from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from langchain_core.runnables import Runnable, RunnableConfig

TState = TypeVar('TState')
TOutput = TypeVar('TOutput')
TRunnableInput = TypeVar('TRunnableInput')
TRunnableOutput = TypeVar('TRunnableOutput')


class BaseNode(
    Runnable[TState, TOutput], Generic[TState, TOutput, TRunnableInput, TRunnableOutput], ABC
):
    def __init__(self, runnable: Runnable[TRunnableInput, TRunnableOutput]):
        self._runnable = runnable

    def invoke(self, input: TState, config: Optional[RunnableConfig] = None) -> TOutput:
        output: TRunnableOutput = self._runnable.invoke(self._build_input(input), config)
        return self._parse_output(state=input, output=output)

    @abstractmethod
    def _build_input(self, state: TState) -> TRunnableInput:
        pass

    @abstractmethod
    def _parse_output(self, state: TState, output: TRunnableOutput) -> TOutput:
        pass
