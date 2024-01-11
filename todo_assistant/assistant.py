from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable


class TODOAssistant:
    def __init__(self, agent: Runnable, max_steps: int = 10) -> None:
        self._agent = agent
        self._conversation_history: list[BaseMessage] = []
        self._max_steps = max_steps

    def step(self) -> dict:
        return self._agent.invoke({"messages": self._conversation_history})

    def add_human_input(self, human_input: str) -> None:
        self._conversation_history.append(HumanMessage(content=human_input))

    def run(self):
        cnt = 0
        while cnt != self._max_steps:
            ai_response = self.step()
            for message in ai_response['messages']:
                if isinstance(message, AIMessage) and message.content:
                    print(message.content)

            print("=" * 10)
            human_input = input("Your response: ")
            self._reset_state()
            self.add_human_input(human_input)
            print("=" * 10)
            cnt += 1

        print("Maximum number of turns reached - ending the conversation.")

    def _reset_state(self) -> None:
        self._conversation_history = []
