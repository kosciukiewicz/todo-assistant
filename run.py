from dependency_injector.wiring import Provide, inject
from langchain.globals import set_debug, set_verbose

from todo_assistant.assistant import TODOAssistant
from todo_assistant.di_containers.application import Application
from todo_assistant.settings import Settings


@inject
def _run(
    todo_assistant: TODOAssistant = Provide[Application.todo_assistant], max_steps: int = 10
) -> None:
    current_step = 0
    ai_response = todo_assistant.step()
    print(ai_response.content)
    print("=" * 10)

    while current_step <= max_steps:
        current_step += 1

        user_input = input("Your response: ")
        print("=" * 10)

        todo_assistant.add_human_input(user_input)
        ai_response = todo_assistant.step()

        print(ai_response.content)
        print("=" * 10)

        if ai_response.is_final_response:
            return

    print("Maximum number of turns reached - ending the conversation.")


if __name__ == '__main__':
    application = Application()
    application.config.from_pydantic(Settings())
    application.init_resources()
    application.wire(modules=[__name__])

    set_debug(application.config.DEBUG())
    set_verbose(application.config.VERBOSE())
    _run(
        max_steps=application.config.MAX_STEPS(),
    )
