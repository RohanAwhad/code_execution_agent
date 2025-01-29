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
    args = function_call.input  # Updated to use ToolUseBlock input directly
    code = args.get('code', '')
    display_code = f'```python\n{code}\n```'
    
    code_result = execute_code_in_notebook(code, kernel_client)
    tool_response = {"role": "tool", "tool_call_id": function_call.id, "content": ""}
    user_messages = []

    for output in code_result:
      if isinstance(output, dict) and output.get('type') == 'image_url':
        user_messages.append(Message('user', [output]))
      else:
        tool_response["content"] += str(output) + "\n"

    if not tool_response["content"].strip():
      tool_response["content"] = "No textual output from execution."

    return {
      'code': code,
      'display_code': display_code,
      'tool_response': tool_response,
      'user_messages': user_messages
    }

  @staticmethod
  def handle_search(function_call: Any) -> Dict:
    args = function_call.input  # Updated to use ToolUseBlock input directly
    try:
      results = search_brave(args.get('query', ''))
      content = "\n\n".join(str(r) for r in results)
      sources = [r.url for r in results]
    except Exception as e:
      content = str(e)
      sources = []

    return {
      'tool_response': {
        "role": "tool",
        "tool_call_id": function_call.id,
        "content": content
      },
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
        history.append({
          "role": msg.role,
          "content": msg.content
        })
      else:
        history.append(msg)

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
  if function_call.name == "execute_code_in_notebook":
    result = ToolHandler.handle_code(function_call, kernel_client)
    messages.append(Message('assistant', result['display_code'], collapsible=True))
    messages[-1].content += f'\n```bash\n{result["tool_response"]["content"]}\n```'
    messages.extend(result['user_messages'])
    
    gpt_messages.append(message)
    gpt_messages.append(result['tool_response'])
    gpt_messages.extend([{"role": m.role, "content": m.content} for m in result['user_messages']])
    
    return {
      'code': result['code'],
      'display_code': result['display_code'],
      'output': f'```bash\n{result["tool_response"]["content"]}\n```',
      'images': result['user_messages']
    }

  elif function_call.name == "search_brave":
    result = ToolHandler.handle_search(function_call)
    gpt_messages.append(message)
    gpt_messages.append(result['tool_response'])
    return {'search_sources': result['search_sources']}

  return None