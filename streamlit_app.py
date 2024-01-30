import streamlit as st

from todo_assistant.di_containers.application import Application
from todo_assistant.settings import Settings

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
            MODEL_NAME="gpt-3.5-turbo-1106",
            MODEL_VERBOSE=False,
        )
        application = Application()
        application.config.from_pydantic(settings)
        application.init_resources()
        application.wire(modules=[__name__])

        st.session_state.todo_assistant = application.todo_assistant()
        ai_response = st.session_state.todo_assistant.step()
        st.session_state.is_final = ai_response.is_final_response
        st.session_state.messages.append({"role": "assistant", "content": ai_response.content})

    st.success("Todo assistant initialized")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.is_final:
    if user_message := st.chat_input("Your response"):
        st.session_state.todo_assistant.add_human_input(user_message)
        with st.chat_message("user"):
            st.markdown(user_message)
        st.session_state.messages.append({"role": "user", "content": user_message})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("▌")
            ai_response = st.session_state.todo_assistant.step()
            message_placeholder.markdown(ai_response.content)
            st.session_state.is_final = ai_response.is_final_response
        st.session_state.messages.append({"role": "assistant", "content": ai_response.content})
else:
    st.info("Finished")
