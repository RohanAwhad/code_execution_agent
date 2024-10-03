import pandas as pd
from pypdf import PdfReader
import dataclasses
import streamlit as st
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, ValidationError
import yaml
import os
import copy
import json
import base64
from io import BytesIO
import openai
from PIL import Image
from typing import Any, Type
from jupyter_client.manager import KernelManager


kernel_manager: KernelManager = None
kernel_client = None


def execute_code_in_notebook(code: str) -> list[Any]:
  if not code:
    return []
  print('Code:')
  print(code)

  global kernel_manager, kernel_client
  if kernel_manager is None:
    kernel_manager = KernelManager()
    kernel_manager.start_kernel()

  if kernel_client is None:
    kernel_client = kernel_manager.client()
    kernel_client.start_channels()
    kernel_client.wait_for_ready()
    code = "%matplotlib inline\n\n" + code

  kernel_client.execute(code)
  output_content: str = ""
  outputs: list[Any] = []
  while True:
    try:
      msg: dict[str, Any] = kernel_client.get_iopub_msg(timeout=5)
      if msg['msg_type'] == 'execute_result':
        outputs.append(msg['content']['data']['text/plain'])
      elif msg['msg_type'] == 'display_data':
        if 'image/png' in msg['content']['data']:
          outputs.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + msg['content']['data']['image/png']}})
      elif msg['msg_type'] == 'stream':
        output_content += msg['content']['text']
      elif msg['msg_type'] == 'error':
        outputs.append("\n".join(msg['content']['traceback']))
    except Exception as e:
      print(f"An error occurred: {e}")
      break

  if output_content:
    outputs.append(output_content)
  print(outputs)
  return outputs


def shutdown_kernel() -> None:
  global kernel_manager
  if kernel_manager is not None:
    kernel_manager.shutdown_kernel()
    kernel_manager = None


# ===
# LLM
# ===
tools = [{
    "type": "function",
    "function": {
        "name": "execute_code_in_notebook",
        "description": "Execute Python code in a Jupyter notebook environment. The notebook state is preserved for the entire session of conversation so you can call previous function without defining them again.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python code to be executed."}},
                "required": ["code"]
        }
    }
}]


@dataclasses.dataclass
class Message:
  role: str
  content: str | list[dict[str, str | dict[str, str]]]


def llm_call_with_tools(model: str, messages: list[Message]) -> Any:  # finding openai chat completion object is crazy
  history = []
  for msg in messages:
    if dataclasses.is_dataclass(msg):
      history.append(dataclasses.asdict(msg))
    else:
      history.append(msg)
  client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
  return client.chat.completions.create(
      model=model,
      messages=history,
      tools=tools,
      tool_choice="auto",
      temperature=0.8,
      max_tokens=4096
  )


# ===
# Streamlit
# ===
def handle_file_upload(uploaded_file) -> None:
  if uploaded_file is not None:
    if 'uploaded_filename' in st.session_state and st.session_state.uploaded_filename == uploaded_file.name:
      return

    st.session_state['uploaded_filename'] = uploaded_file.name
    print('this function was called')
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension in ['.png', '.jpg', '.jpeg']:
      file_bytes = uploaded_file.read()
      image_data = base64.b64encode(file_bytes).decode('utf-8')
      thumbnail = Image.open(BytesIO(file_bytes))
      #st.image(thumbnail, caption='Uploaded Image', use_column_width=True)
      st.session_state.messages.append(Message(role="user", content=[{'type': 'image_url', 'image_url': {
                                       'url': f'data:image/{file_extension[1:]};base64,' + image_data}}]))
      st.session_state.gpt_messages.append(Message(role="user", content=[{'type': 'image_url', 'image_url': {
                                           'url': f'data:image/{file_extension[1:]};base64,' + image_data}}]))

    elif file_extension == '.pdf':
      pdf_reader = PdfReader(uploaded_file)
      text_content = ""
      for page in pdf_reader.pages:
        text_content += page.extract_text() + "\n"
      st.session_state.messages.append(Message(role="user", content=text_content))
      st.session_state.gpt_messages.append(Message(role="user", content=text_content))

    elif file_extension in ['.csv', '.xls', '.xlsx']:
      file_path = f"./data/{uploaded_file.name}"
      with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
      st.session_state.messages.append(Message(role="user", content=f"Stored {uploaded_file.name} in the ./data directory."))
      st.session_state.gpt_messages.append(Message(role="user", content=f"Stored {uploaded_file.name} in the ./data directory."))

    else:
      file_path = f"./data/{uploaded_file.name}"
      with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
      st.session_state.messages.append(Message(role="user", content=f"Stored {uploaded_file.name} in the ./data directory."))
      st.session_state.gpt_messages.append(Message(role="user", content=f"Stored {uploaded_file.name} in the ./data directory."))


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
    st.session_state.gpt_messages = copy.deepcopy(st.session_state.global_messages[key])


def create_messaging_window() -> None:
  global kernel_client, kernel_manager
  st.title("Chat with AI and Code Execution")

  if 'global_messages' not in st.session_state:
    st.session_state.global_messages: Dict[str, List[Message]] = load_global_messages_from_disk()

  clear_chat_button: bool = st.sidebar.button("Clear Chat")
  if clear_chat_button:
    st.session_state.messages = []
    st.session_state.gpt_messages = []
    if kernel_client is not None:
      kernel_client.stop_channels()
    if kernel_manager is not None:
      kernel_manager.shutdown_kernel()
    kernel_client = None
    kernel_manager = None

  if 'messages' not in st.session_state:
    st.session_state.messages: List[Message] = []

  if 'gpt_messages' not in st.session_state:
    st.session_state.gpt_messages: List[Any] = []

  st.sidebar.write("Previous conversations:")
  for key in st.session_state.global_messages.keys():
    btn_name: str = key if len(key) < 20 else f'{key[:17]} ...'
    btn_name = btn_name.ljust(21)
    if st.sidebar.button(btn_name, key=key, on_click=load_conversation, args=(key,)):
      pass

  for message in st.session_state.messages:
    if dataclasses.is_dataclass(message):
      with st.chat_message(message.role):
        if isinstance(message.content, str):
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

  uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "pdf", "csv", "xls", "xlsx"])
  handle_file_upload(uploaded_file)

  user_input: str = st.chat_input("Type your message here...")
  if user_input:
    user_message_content = user_input
    user_message = Message(role="user", content=user_message_content)
    st.session_state.messages.append(user_message)
    st.session_state.gpt_messages.append(user_message)

    if st.session_state.messages:
      for msg in st.session_state.messages:
        if isinstance(msg.content, str):
          first_message_content: str = msg.content
          break

    with st.chat_message("user"):
      st.write(user_message_content)

    for _ in range(3):
      ai_response = llm_call_with_tools("gpt-4o-2024-08-06", st.session_state.gpt_messages)
      assistant_message = ai_response.choices[0].message
      if assistant_message.tool_calls:
        st.session_state.gpt_messages.append(assistant_message)
        for function_call in assistant_message.tool_calls:
          if function_call.function.name == "execute_code_in_notebook":
            args = json.loads(function_call.function.arguments)
            code_result = execute_code_in_notebook(args.get('code', ''))
            # Prepare tool response based on the result
            tool_call_response = {"role": "tool", "tool_call_id": function_call.id, "content": ""}
            user_messages = []
            for output in code_result:
              if isinstance(output, dict) and 'type' in output and output['type'] == 'image_url':
                user_messages.append(Message('user', [output]))
              else:
                tool_call_response["content"] += str(output) + "\n"
            # If no non-image output, still add an empty tool call response
            if not tool_call_response["content"].strip:
              tool_call_response["content"] = "No textual output from execution."

            st.session_state.gpt_messages.extend([tool_call_response] + user_messages)
            st.session_state.messages.extend(user_messages)
            if user_messages:
              for msg in user_messages:
                for item in msg.content:
                  image_data = base64.b64decode(item['image_url']['url'].split(",")[1])
                  image = Image.open(BytesIO(image_data))
                  with st.chat_message('user'):  # this is originally tool
                    st.image(image)
      else:
        st.session_state.messages.append(Message(role="assistant", content=assistant_message.content))
        st.session_state.gpt_messages.append(Message(role="assistant", content=assistant_message.content))
        with st.chat_message("assistant"):
          st.write(assistant_message.content)
        break

    st.session_state.global_messages[first_message_content] = st.session_state.messages
    save_global_messages_to_disk(st.session_state.global_messages)


if __name__ == "__main__":
  try:
    create_messaging_window()
  except Exception as e:
    print(e)
  finally:
    try:
      if kernel_client is not None:
        kernel_client.stop_channels()
    except Exception as e:
      print(e)
    try:
      if kernel_manager is not None:
        kernel_manager.shutdown_kernel()
    except Exception as e:
      print(e)
