from dependency_injector.wiring import Provide, inject

from todo_assistant.assistant import TODOAssistant
from todo_assistant.di_containers.application import Application


@inject
def main(
    todo_assistant: TODOAssistant = Provide[Application.todo_assistant],
) -> None:
    return todo_assistant.run()


if __name__ == '__main__':
    application = Application()
    application.init_resources()
    application.wire(modules=[__name__])
    main()
