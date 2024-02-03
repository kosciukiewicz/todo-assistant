import asyncio

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from todo_assistant.assistant import BaseAssistantResponseCallback, TODOAssistantResponse
from todo_assistant.di_containers.application import Application
from todo_assistant.settings import Settings


def mark_assistant_processing():
    st.session_state["assistant_processing"] = True


def mark_assistant_ready():
    st.session_state["assistant_processing"] = False


class StreamlitAssistantResponseCallback(BaseAssistantResponseCallback):
    def __init__(self, message_placeholder: DeltaGenerator):
        self._full_response = ''
        self._message_placeholder = message_placeholder

    def on_stream_new_token(self, token: str) -> None:
        self._full_response += token
        self._message_placeholder.markdown(self._full_response + "▌")

    def on_stream_finish(self) -> None:
        pass

    def on_response(self, final_response: TODOAssistantResponse) -> None:
        st.session_state.messages.append({"role": "assistant", "content": final_response.content})
        st.session_state.is_final = final_response.is_final_response
        mark_assistant_ready()
        st.rerun()


async def main():
    st.title("🧠 TODO Assistant")
    st.subheader("📝 A Notion TODO Board assistant powered by OpenAI LLM")

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    notion_api_key = st.sidebar.text_input("Notion API Key", type="password")
    notion_database_id = st.sidebar.text_input("Notion Database ID", type="default")

    if not openai_api_key:
        st.info("Please enter your OpenAI API key!", icon="⚠")
        st.stop()

    if not notion_api_key:
        st.info("Please enter your NOTION API key!", icon="⚠")
        st.stop()

    if not notion_database_id:
        st.info("Please enter your Database ID!", icon="⚠")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.is_final = False

    if ("todo_assistant" not in st.session_state) and all(
        (openai_api_key, notion_api_key, notion_database_id)
    ):
        with st.spinner("Todo assistant initializing"):
            settings = Settings(
                OPENAI_API_KEY=openai_api_key,
                NOTION_API_KEY=notion_api_key,
                NOTION_DATABASE_ID=notion_database_id,
            )
            application = Application()
            application.config.from_pydantic(settings)
            application.init_resources()
            application.wire(modules=[__name__])

            st.session_state.todo_assistant = application.todo_assistant()
            ai_response = st.session_state.todo_assistant.step()
            st.session_state.is_final = ai_response.is_final_response
            st.session_state.assistant_processing = False
            st.session_state.messages.append({"role": "assistant", "content": ai_response.content})

        st.success("Todo assistant initialized")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.is_final:
        if user_message := st.chat_input(
            "Your response",
            disabled=st.session_state.assistant_processing,
            on_submit=mark_assistant_processing,
        ):
            st.session_state.todo_assistant.add_human_input(user_message)

            with st.chat_message("user"):
                st.markdown(user_message)
            st.session_state.messages.append({"role": "user", "content": user_message})

            with st.chat_message("assistant"):
                message_placeholder = st.empty().markdown("▌")
                await st.session_state.todo_assistant.astep(
                    new_token_callback=StreamlitAssistantResponseCallback(
                        message_placeholder=message_placeholder
                    )
                )
    else:
        st.info("Finished")


if __name__ == '__main__':
    asyncio.run(main())
