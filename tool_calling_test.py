import jupyter_client
import json
import openai
import time

from typing import Any, Dict, List
from jupyter_client.manager import KernelManager

# Global variable to hold the kernel manager
kernel_manager: KernelManager = None


def execute_code_in_notebook(code: str) -> list[Any]:
  if not code:
    return []

  global kernel_manager
  # Create a kernel if it doesn't exist
  if kernel_manager is None:
    kernel_manager = KernelManager()
    kernel_manager.start_kernel()

  # Create a client for the kernel
  kernel_client = kernel_manager.client()
  kernel_client.start_channels()
  kernel_client.wait_for_ready()
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
  return outputs


def shutdown_kernel() -> None:
  global kernel_manager
  if kernel_manager is not None:
    kernel_manager.shutdown_kernel()
    kernel_manager = None


# ===
# LLM
# ===
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_code_in_notebook",
            "description": "Execute Python code in a Jupyter notebook environment. The notebook state is preserved for the entire session of conversation so you can call previous function without defining them again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to be executed."
                    }
                },
                "required": ["code"]
            }
        }
    }
]
client = openai.OpenAI()
messages = [{"role": "system", "content": "You are a helpful assistant that can execute Python code in a Jupyter notebook environment."}]
while True:
  try:
    user_input = input('Enter your message:')
    messages.append({'role': 'user', 'content': user_input})
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # Process the response
    assistant_message = response.choices[0].message
    messages.append(assistant_message)
    print(assistant_message)
    if assistant_message.tool_calls:
      for function_call in assistant_message.tool_calls:
        if function_call.function.name == "execute_code_in_notebook":
          args = json.loads(function_call.function.arguments)
          code_result = execute_code_in_notebook(args.get('code', ''))

          # Prepare tool response based on the result
          tool_call_response = {"role": "tool", "tool_call_id": function_call.id, "content": ""}
          user_messages = []
          for output in code_result:
            if isinstance(output, dict) and 'type' in output and output['type'] == 'image_url':
              user_messages.append({'role': 'user', 'content': [output]})
            else:
              tool_call_response["content"] += str(output) + "\n"

          # If no non-image output, still add an empty tool call response
          if not tool_call_response["content"].strip():
            tool_call_response["content"] = "No textual output from execution."
          messages.extend([tool_call_response] + user_messages)

      # Make another API call to process the function result
      second_response = client.chat.completions.create(
          model="gpt-4o-2024-08-06",
          messages=messages
      )
      print(second_response.choices[0].message.content)
      messages.append(second_response.choices[0].message)
    else:
      print(assistant_message.content)
  except KeyboardInterrupt:
    break

shutdown_kernel()
