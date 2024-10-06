import atexit
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
import requests


def execute_code_in_notebook(code: str, kernel_manager: KernelManager, kernel_client) -> list[Any]:
  if not code:
    return []
  print('Code:')
  print(code)
  code = "%matplotlib inline\n\n" + code
  kernel_client.execute(code)
  output_content: str = ""
  outputs: list[Any] = []
  while True:
    try:
      msg: dict[str, Any] = kernel_client.get_iopub_msg(timeout=30)
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


def shutdown_kernel(kernel_manager, kernel_client) -> None:
  if kernel_client is not None:
    kernel_client.stop_channels()
  if kernel_manager is not None:
    kernel_manager.shutdown_kernel()


# ===
# Brave Search
# ===

@dataclasses.dataclass
class SearchResult:
  """
  Dataclass to represent the search results from Brave Search API.

  :param title: The title of the search result.
  :param url: The URL of the search result.
  :param description: A brief description of the search result.
  :param extra_snippets: Additional snippets related to the search result.
  """
  title: str
  url: str
  description: str
  extra_snippets: list

  def __str__(self) -> str:
    """
    Returns a string representation of the search result.

    :return: A string representation of the search result.
    """
    return (
        f"Title: {self.title}\n"
        f"URL: {self.url}\n"
        f"Description: {self.description}\n"
        f"Extra Snippets: {', '.join(self.extra_snippets)}"
    )


def search_brave(query: str, count: int = 10) -> List[SearchResult]:
  """
  Searches the web using Brave Search API and returns structured search results.

  :param query: The search query string.
  :param count: The number of search results to return.
  :return: A list of SearchResult objects containing the search results.
  """
  if not query:
    return []
  url = "https://api.search.brave.com/res/v1/web/search"
  headers = {
      "Accept": "application/json",
      "X-Subscription-Token": os.environ['BRAVE_SEARCH_AI_API_KEY']
  }
  params = {
      "q": query,
      "count": count
  }

  response = requests.get(url, headers=headers, params=params)
  response.raise_for_status()  # Raises an exception for HTTP errors
  results_json = response.json()

  results = []
  for item in results_json.get('web', {}).get('results', []):
    result = SearchResult(
        title=item.get('title', ''),
        url=item.get('url', ''),
        description=item.get('description', ''),
        extra_snippets=item.get('extra_snippets', [])
    )
    results.append(result)

  print('Search results')
  print(results)
  return results


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
}, {
    "type": "function",
    "function": {
        "name": "search_brave",
        "description": "Search the web using Brave Search API and returns structured search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query string."},
            },
            "required": ["query"]
        }
    }
}]


@dataclasses.dataclass
class Message:
  role: str
  content: str | list[dict[str, str | dict[str, str]]]
  collapsible: bool = False


def llm_call_with_tools(model: str, messages: list[Message]) -> Any:  # finding openai chat completion object is crazy
  history = []
  for msg in messages:
    if dataclasses.is_dataclass(msg):
      history.append(dataclasses.asdict(msg))
    else:
      history.append(msg)

  print('Latest message to llm:')
  print(history[-1])
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
  st.title("Chat with AI and Code Execution")

  if 'global_messages' not in st.session_state:
    st.session_state.global_messages: Dict[str, List[Message]] = load_global_messages_from_disk()

  clear_chat_button: bool = st.sidebar.button("Clear Chat")
  if clear_chat_button:
    st.session_state.messages = []
    st.session_state.gpt_messages = []
    shutdown_kernel(st.session_state.kernel_manager, st.session_state.kernel_client)
    st.session_state.kernel_manager = None
    st.session_state.kernel_client = None

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

  uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "pdf", "csv", "xls", "xlsx"])
  handle_file_upload(uploaded_file)

  user_input: str = st.chat_input("Type your message here...")
  if user_input:
    search_sources = []
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

    while True:
      with st.spinner('Calling AI ...'):
        ai_response = llm_call_with_tools("gpt-4o-2024-08-06", st.session_state.gpt_messages)
      assistant_message = ai_response.choices[0].message
      if assistant_message.tool_calls:
        st.session_state.gpt_messages.append(assistant_message)
        for function_call in assistant_message.tool_calls:
          args = json.loads(function_call.function.arguments)
          if function_call.function.name == "execute_code_in_notebook":
            code = args.get('code', '')
            display_code = f'```python\n{code}\n```'
            st.session_state.messages.append(Message('assistant', display_code, collapsible=True))
            with st.spinner('Executing Code ...'):
              with st.expander("Click to see code"):
                st.write(display_code)
              code_result = execute_code_in_notebook(code, st.session_state.kernel_manager, st.session_state.kernel_client)
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
          elif function_call.function.name == "search_brave":
            with st.spinner('Searching Internet ...'):
              try:
                search_results = search_brave(args.get('query', ''), args.get('count', 10))
              except Exception as e:
                search_results = e
            # Prepare tool response based on the search results
            tool_call_response = {"role": "tool", "tool_call_id": function_call.id, "content": ""}
            search_sources = []
            if isinstance(search_results, list):
              for result in search_results:
                tool_call_response["content"] += str(result) + "\n\n"
                search_sources.append(result.url)  # Collect URLs
            else:
              tool_call_response['content'] = search_results
            st.session_state.gpt_messages.append(tool_call_response)
      else:
        st.session_state.messages.append(Message(role="assistant", content=assistant_message.content))
        st.session_state.gpt_messages.append(Message(role="assistant", content=assistant_message.content))
        if len(search_sources):
          ret = "\n\nSource URLs:"
          for url in search_sources:
            ret += f"\n- {url}"
          search_sources = []
          st.session_state.messages[-1].content += ret
        with st.chat_message("assistant"):
          st.write(st.session_state.messages[-1].content)
        break

    st.session_state.global_messages[first_message_content] = st.session_state.messages
    save_global_messages_to_disk(st.session_state.global_messages)


create_messaging_window()
