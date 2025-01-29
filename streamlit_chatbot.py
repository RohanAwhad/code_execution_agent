import atexit
import dataclasses
import streamlit as st
import os
import copy
import base64
from io import BytesIO
from PIL import Image
from jupyter_client.manager import KernelManager
import time

from src.tools.code_executor import shutdown_kernel
from src.llm import Message, llm_call_with_tools, handle_tool_calls
from src.utils import load_global_messages_from_disk, save_global_messages_to_disk, process_uploaded_file

def on_shutdown():
  shutdown_kernel(st.session_state.kernel_manager, st.session_state.kernel_client)
atexit.register(on_shutdown)

if 'kernel_manager' not in st.session_state or st.session_state.kernel_manager is None:
  st.session_state.kernel_manager = KernelManager()
  st.session_state.kernel_manager.start_kernel()
if 'kernel_client' not in st.session_state or st.session_state.kernel_client is None:
  st.session_state.kernel_client = st.session_state.kernel_manager.client()
  st.session_state.kernel_client.start_channels()
  st.session_state.kernel_client.wait_for_ready()

def handle_file_upload(uploaded_file) -> None:
  if uploaded_file is not None:
    if 'uploaded_filename' in st.session_state and st.session_state.uploaded_filename == uploaded_file.name:
      return

    st.session_state['uploaded_filename'] = uploaded_file.name
    message_content, thumbnail = process_uploaded_file(uploaded_file, uploaded_file.name)
    
    if thumbnail:
      st.image(thumbnail, caption='Uploaded Image', use_column_width=True)
      
    st.session_state.messages.append(Message(role="user", content=message_content))
    st.session_state.gpt_messages.append(Message(role="user", content=message_content))


def load_conversation(key: str) -> None:
  if key in st.session_state.global_messages:
    st.session_state.messages = st.session_state.global_messages[key]
    st.session_state.gpt_messages = copy.deepcopy(st.session_state.global_messages[key])
    st.session_state.chat_session_key = key


def add_clear_chat_button() -> None:
  clear_chat_button: bool = st.sidebar.button("Clear Chat")
  if clear_chat_button:
    st.session_state.messages = []
    st.session_state.gpt_messages = []
    shutdown_kernel(st.session_state.kernel_manager, st.session_state.kernel_client)
    st.session_state.kernel_manager = None
    st.session_state.kernel_client = None
    st.session_state.chat_session_key = None


def let_user_add_env_keys() -> None:
  st.sidebar.header("Environment Variables")
  new_key: str = st.sidebar.text_input("Key")
  new_value: str = st.sidebar.text_input("Value", type="password")

  if 'env_keys' not in st.session_state:
    st.session_state.env_keys = {}

  if st.sidebar.button("Add Key"):
    if new_key and new_value:
      os.environ[new_key] = new_value
      st.session_state.env_keys[new_key] = new_value
      st.sidebar.success(f"Added {new_key}")
    else:
      st.sidebar.error("Please provide both key and value")

  for key in st.session_state.env_keys.keys():
    masked_value: str = '*' * 5
    st.sidebar.write(f"{key}: {masked_value}")


def list_prev_conv_threads() -> None:
  st.sidebar.write("Previous conversations:")
  for key in st.session_state.global_messages.keys():
    btn_name: str = key if len(key) < 20 else f'{key[:17]} ...'
    btn_name = btn_name.ljust(21)
    if st.sidebar.button(btn_name, key=key, on_click=load_conversation, args=(key,)):
      pass

def render_message(message: Message) -> None:
  if dataclasses.is_dataclass(message):
    with st.chat_message(message.role):
      if isinstance(message.content, str):
        if message.collapsible:
          with st.expander("Click to see code"):
            st.write(message.content)
        else:
          st.write(message.content)
      elif isinstance(message.content, list):
        for item in message.content:
          if 'type' in item:
            if item['type'] == 'text':
              st.write(item['content'])
            elif item['type'] == 'image_url':
              image_data = base64.b64decode(item['image_url']['url'].split(",")[1])
              image = Image.open(BytesIO(image_data))
              st.image(image)

def create_messaging_window() -> None:
  st.title("Chat with AI and Code Execution")

  if 'global_messages' not in st.session_state: st.session_state.global_messages = load_global_messages_from_disk()
  if 'messages' not in st.session_state: st.session_state.messages = []
  if 'gpt_messages' not in st.session_state: st.session_state.gpt_messages = []

  let_user_add_env_keys()
  add_clear_chat_button()
  list_prev_conv_threads()
  for message in st.session_state.messages: render_message(message)

  uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "pdf", "csv", "xls", "xlsx", 'txt'])
  handle_file_upload(uploaded_file)

  user_input: str = st.chat_input("Type your message here...")
  if user_input:
    search_sources = []
    user_message = Message(role="user", content=user_input)
    st.session_state.messages.append(user_message)
    st.session_state.gpt_messages.append(user_message)

    if ('chat_session_key' not in st.session_state or st.session_state.chat_session_key is None) and st.session_state.messages:
      for msg in st.session_state.messages:
        if isinstance(msg.content, str):
          st.session_state.chat_session_key = msg.content + str(int(time.time()))
          break

    with st.chat_message("user"):
      st.write(user_input)

    while True:
      with st.spinner('Calling AI ...'):
        ai_response = llm_call_with_tools("claude-3-5-sonnet-20241022", st.session_state.gpt_messages)
      assistant_message = ai_response.choices[0].message
      result = handle_tool_calls(assistant_message, st.session_state.gpt_messages, st.session_state.messages, st.session_state.kernel_client)
      if not result: continue

      if 'code' in result:
        with st.spinner('Executing Code ...'):
          with st.expander("Click to see code"):
            st.write(result['display_code'])
          with st.expander("See Output"):
            st.write(result['output'])
          if 'images' in result and result['images']:
            for msg in result['images']:
              for item in msg.content:
                image_data = base64.b64decode(item['image_url']['url'].split(",")[1])
                image = Image.open(BytesIO(image_data))
                with st.chat_message('user'):
                  st.image(image)
                  
      elif 'search_sources' in result:
        search_sources.extend(result['search_sources'])
        
      elif 'final_response' in result:
        response_content = result['final_response']
        if search_sources:
          response_content += "\n\nSource URLs:"
          for url in search_sources:
            response_content += f"\n- {url}"
        with st.chat_message("assistant"):
          st.write(response_content)
        break

    st.session_state.global_messages[st.session_state.chat_session_key] = st.session_state.messages
    save_global_messages_to_disk(st.session_state.global_messages)

create_messaging_window()
