import dataclasses
import os
import json
import logging
from typing import Any, Dict, List
from anthropic import Anthropic

from src.tools.code_executor import execute_code_in_notebook
from src.tools.web_search import search_brave

# Config
TOOLS = [
    {
        "name": "execute_code_in_notebook",
        "description": "Execute Python code in a Jupyter notebook environment.",
        "input_schema": {
            "type": "object", 
            "properties": {"code": {"type": "string", "description": "Python code"}},
            "required": ["code"]
        }
    },
    {
        "name": "search_brave",
        "description": "Search using Brave Search API",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]
        }
    }
]

SYSTEM_PROMPT = '''
You are a language model. Your task is to answer users queries. You have access to internet search and python code execution.

Search internet when the user asks, or you are stuck in debugging the python code. Always write small chunks of code so that they are quickly executed and you can iterate on the errors fast. You can install packages on the system, pip and apt packages.

The environment is only CPU-based, so keep that in mind. All the data related to computation given from the user will be kept in `./data` dir, and you should output also in that dir only.

Never use tqdm.
'''.strip()

@dataclasses.dataclass
class Message:
  role: str
  content: str | List[Dict[str, str]]
  collapsible: bool = False

class ToolHandler:
  @staticmethod
  def handle_code(function_call: Any, kernel_client: Any) -> Dict:
    args = function_call.input
    code = args.get('code', '')
    display_code = f'```python\n{code}\n```'
    
    code_result = execute_code_in_notebook(code, kernel_client)
    tool_output = ""
    user_messages = []

    for output in code_result:
      if isinstance(output, dict) and output.get('type') == 'image_url':
        user_messages.append(Message('user', [output]))
      else:
        tool_output += str(output) + "\n"

    if not tool_output.strip():
      tool_output = "No textual output from execution."

    return {
      'code': code,
      'display_code': display_code,
      'tool_output': tool_output,
      'user_messages': user_messages
    }

  @staticmethod
  def handle_search(function_call: Any) -> Dict:
    args = function_call.input
    try:
      results = search_brave(args.get('query', ''))
      content = "\n\n".join(str(r) for r in results)
      sources = [r.url for r in results]
    except Exception as e:
      content = str(e)
      sources = []

    return {
      'tool_output': content,
      'search_sources': sources
    }

class LLMHandler:
  def __init__(self, model: str):
    self.model = model
    self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    self.tool_handler = ToolHandler()
    
  def call(self, messages: List[Message]) -> Any:
    history = []
    for msg in messages:
      if dataclasses.is_dataclass(msg):
        if msg.role in ['user', 'assistant']:
          content = msg.content
          if isinstance(content, str):
            content = content.strip()
          history.append({
            "role": msg.role,
            "content": content
          })
      else:
        if msg.get('role') in ['user', 'assistant']:
          content = msg.get('content', '')
          if isinstance(content, str):
            content = content.strip()
          history.append({
            "role": msg['role'],
            "content": content
          })

    logging.info(f'Latest message to LLM: {history[-1]}')
    
    return self.client.messages.create(
      model=self.model,
      system=SYSTEM_PROMPT,
      messages=history,
      tools=TOOLS,
      max_tokens=4096,
      temperature=0.8
    )

def llm_call_with_tools(model: str, messages: List[Message]) -> Any:
  handler = LLMHandler(model)
  return handler.call(messages)


def handle_tool_calls(message: Any, gpt_messages: List, messages: List, kernel_client: Any) -> Dict | None:
  tool_uses = [c for c in message.content if c.type == "tool_use"]
  if not tool_uses:
    messages.append(Message(role="assistant", content=message.content[0].text))
    gpt_messages.append({"role": "assistant", "content": message.content[0].text})
    return {'final_response': message.content[0].text}

  function_call = tool_uses[0]
  tool_result_content = []
  result_data = {}

  if function_call.name == "execute_code_in_notebook":
    result = ToolHandler.handle_code(function_call, kernel_client)
    result_data = {
      'code': result['code'],
      'display_code': result['display_code'],
      'output': f'```bash\n{result["tool_output"]}\n```',
      'images': result['user_messages']
    }
    
    # Build proper tool result content
    tool_result_content.append({"type": "text", "text": result["tool_output"]})
    for img_msg in result['user_messages']:
      tool_result_content.extend(img_msg.content)

  elif function_call.name == "search_brave":
    result = ToolHandler.handle_search(function_call)
    result_data = {'search_sources': result['search_sources']}
    tool_result_content.append({"type": "text", "text": result['tool_output']})

  # Create proper tool result message
  tool_result_msg = {
    "role": "user",
    "content": [{
      "type": "tool_result",
      "tool_use_id": function_call.id,
      "content": tool_result_content
    }]
  }

  # Append to both message lists
  messages.append(Message(role="user", content=tool_result_msg["content"]))
  gpt_messages.append(message.model_dump())
  gpt_messages.append(tool_result_msg)
  return result_data
