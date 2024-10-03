import dataclasses
import openai
import os
import json
import yaml

from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, List

import streamlit as st


@dataclasses.dataclass
class Message:
    role: str
    content: str

def llm_call(model: str, messages: List[Message]) -> str:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    res = client.chat.completions.create(model=model, messages=[dataclasses.asdict(x) for x in messages], temperature=0.8, max_tokens=4096)
    return res.choices[0].message.content


def load_global_messages_from_disk() -> Dict[str, List[Message]]:
    if os.path.exists("global_messages.json"):
        with open("global_messages.json", "r") as file:
            data = json.load(file)
            return {k: [Message(**msg) for msg in v] for k, v in data.items()}
    else:
        with open("global_messages.json", "w") as file:
            json.dump({}, file)
        return {}

def save_global_messages_to_disk(global_messages: Dict[str, List[Message]]) -> None:
    with open("global_messages.json", "w") as file:
        json.dump({k: [dataclasses.asdict(msg) for msg in v] for k, v in global_messages.items()}, file)

def load_conversation(key: str) -> None:
    if key in st.session_state.global_messages:
        st.session_state.messages = st.session_state.global_messages[key]

def create_messaging_window() -> None:
    st.title("Chat with AI")

    if 'global_messages' not in st.session_state:
        st.session_state.global_messages: Dict[str, List[Message]] = load_global_messages_from_disk()

    clear_chat_button: bool = st.sidebar.button("Clear Chat")
    if clear_chat_button:
        st.session_state.messages = []

    # Initialize session state for messages if not already present
    if 'messages' not in st.session_state:
        st.session_state.messages: List[Message] = []

    # List all keys in GLOBAL_MESSAGES in the sidebar
    st.sidebar.write("Previous conversations:")
    for key in st.session_state.global_messages.keys():
        if st.sidebar.button(key, key=key, on_click=load_conversation, args=(key,)):
            pass

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message.role):
            st.write(message.content)

    # File uploader
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "doc", "docx"])

    # User input
    user_input: str = st.chat_input("Type your message here...")

    if user_input or uploaded_file:
        # Process user input
        if user_input:
            user_message_content = user_input
        else:
            user_message_content = "File uploaded: " + uploaded_file.name

        # Add user message to the conversation
        user_message = Message(role="user", content=user_message_content)
        st.session_state.messages.append(user_message)

        # Update session state 'global_messages' with the current conversation
        if st.session_state.messages:
            first_message_content: str = st.session_state.messages[0].content
            st.session_state.global_messages[first_message_content] = st.session_state.messages
            save_global_messages_to_disk(st.session_state.global_messages)

        # Display user message
        with st.chat_message("user"):
            st.write(user_message_content)

        # If a file was uploaded, you might want to process it here
        if uploaded_file:
            # Read and process the file content
            file_contents = uploaded_file.read()
            # You can add your file processing logic here
            # For example, you could add the file contents to the message
            st.session_state.messages.append(Message(role="user", content=f"File contents: {file_contents}"))

        # Generate AI response
        ai_response: str = llm_call("gpt-4o-2024-08-06", st.session_state.messages)
        ai_message = Message(role="assistant", content=ai_response)
        st.session_state.messages.append(ai_message)

        # Display AI response
        with st.chat_message("assistant"):
            st.write(ai_response)

# Run the messaging window
if __name__ == "__main__":
    create_messaging_window()

