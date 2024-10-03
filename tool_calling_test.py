import jupyter_client
from jupyter_client.manager import KernelManager
import json
import time
from typing import Dict, Any

# Global variable to hold the kernel manager
kernel_manager: KernelManager = None
def execute_code_in_notebook(code: str) -> list[Any]:
    if not code: return []
    global kernel_manager
    # Create a kernel if it doesn't exist
    if kernel_manager is None:
        kernel_manager = KernelManager()
        kernel_manager.start_kernel()
    # Create a client for the kernel
    kernel_client = kernel_manager.client()
    kernel_client.start_channels()
    # Ensure that execution state is idle before executing code
    kernel_client.wait_for_ready()
    # Execute the code in the kernel
    kernel_client.execute(code)
    output_content: str = ""
    outputs: list[Any] = []

    while True:
        try:
            msg: dict[str, Any] = kernel_client.get_iopub_msg(timeout=5)
            if msg['msg_type'] == 'execute_result':
                # Handle execute_result messages for text output
                outputs.append(msg['content']['data']['text/plain'])
            elif msg['msg_type'] == 'display_data':
                # Handle display_data messages for rich output like images
                if 'image/png' in msg['content']['data']:
                    outputs.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + msg['content']['data']['image/png']}})
            elif msg['msg_type'] == 'stream':
                # Handle stream messages for stdout
                output_content += msg['content']['text']
            elif msg['msg_type'] == 'error':
                # Handle error messages
                outputs.append("\n".join(msg['content']['traceback']))

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    if output_content:
        outputs.append(output_content)

    print(outputs)
    return outputs

# Ensure proper cleanup
def shutdown_kernel():
    global kernel_manager
    if kernel_manager is not None:
        kernel_manager.shutdown_kernel()
        kernel_manager = None

import openai
import json
from typing import List, Dict


import base64
from pathlib import Path


# Initialize the OpenAI client
client = openai.OpenAI()

# Define the function for the tool
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

# Create a conversation with the AI
messages = [
    {"role": "system", "content": "You are a helpful assistant that can execute Python code in a Jupyter notebook environment."},
]

# Make the API call
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
            # The model wants to call a function
            for function_call in assistant_message.tool_calls:
                if function_call.function.name == "execute_code_in_notebook":
                    # Call the function and get the result
                    args = json.loads(function_call.function.arguments)
                    code_result = execute_code_in_notebook(args.get('code', ''))
                    
                    # Prepare messages based on the result
                    tool_call_response = {"role": "tool", "tool_call_id": function_call.id, "content": ""}
                    user_messages = []

                    for output in code_result:
                        if isinstance(output, dict) and 'type' in output and output['type'] == 'image_url':
                            user_messages = {'role': 'user', 'content': [output]}
                        else:
                            tool_call_response["content"] += str(output) + "\n"

                    # If no non-image output, still add an empty tool call response
                    if not tool_call_response["content"].strip():
                        tool_call_response["content"] = "No textual output from execution."

                    # Append all parts to the conversation
                    messages.extend([
                        tool_call_response,
                    ] + user_messages)

            # Make another API call to process the function result
            second_response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=messages
            )

            print(second_response.choices[0].message.content)
            messages.append(second_response.choices[0].message)
        else:
            # The model responded without calling a function
            print(assistant_message.content)
    except KeyboardInterrupt:
        break

shutdown_kernel()
