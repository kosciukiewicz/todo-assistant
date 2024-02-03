from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langsmith import traceable
from pydantic import BaseModel

_FINAL_MESSAGE = "Assistant decided to end conversation"
_STOP_INDICATOR = '<FINISH>'


class TODOAssistantResponse(BaseModel):
    content: str
    is_final_response: bool


class TODOAssistant:
    def __init__(self, agent: Runnable, max_steps: int = 10) -> None:
        self._agent = agent
        self._conversation_history: list[BaseMessage] = []
        self._max_steps = max_steps

    @traceable(
        run_type="chain",
        name="Step",
    )
    def step(self) -> TODOAssistantResponse:
        response = self._agent.invoke({"messages": self._conversation_history})
        for message in reversed(response['messages']):
            if isinstance(message, AIMessage) and message.content:
                response = str(message.content)
                if _STOP_INDICATOR in response:
                    response = response.replace(_STOP_INDICATOR, '')
                    return TODOAssistantResponse(
                        content=response or _FINAL_MESSAGE,
                        is_final_response=True,
                    )
                else:
                    return TODOAssistantResponse(
                        content=response,
                        is_final_response=False,
                    )

        return TODOAssistantResponse(
            content=_FINAL_MESSAGE,
            is_final_response=True,
        )

    def add_human_input(self, human_input: str) -> None:
        self._reset_state()
        self._conversation_history.append(HumanMessage(content=human_input))

    def _reset_state(self) -> None:
        self._conversation_history = []
