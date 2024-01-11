from abc import ABC, abstractmethod

from langgraph.pregel import Pregel


class BaseGraphBuilder(ABC):
    @abstractmethod
    def build_graph(self) -> Pregel:
        pass
