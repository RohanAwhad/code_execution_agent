import anthropic
import os

from typing import Any
from jupyter_client.manager import KernelManager

kernel_manager: KernelManager = None

def execute_code_in_notebook(code: str) -> list[Any]:
    if not code:
        return []

    global kernel_manager
    if kernel_manager is None:
        kernel_manager = KernelManager()
        kernel_manager.start_kernel()

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
            elif msg['msg_type'] == 'display_data' and 'image/png' in msg['content']['data']:
                outputs.append({'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{msg['content']['data']['image/png']}"}})
            elif msg['msg_type'] == 'stream':
                output_content += msg['content']['text']
            elif msg['msg_type'] == 'error':
                outputs.append("\n".join(msg['content']['traceback']))
        except Exception as e:
            print(f"Error: {e}")
            break

    if output_content:
        outputs.append(output_content)
    return outputs

def shutdown_kernel() -> None:
    global kernel_manager
    if kernel_manager is not None:
        kernel_manager.shutdown_kernel()
        kernel_manager = None

tools = [
    {
        "name": "execute_code_in_notebook",
        "description": "Execute Python code in Jupyter notebook with persistent state",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    }
]

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
system_prompt = "You are an assistant that executes Python code in a Jupyter notebook environment."
messages = []
last_assistant_message = None

while True:
    try:
        user_input = input('Enter your message: ')
        messages.append({"role": "user", "content": user_input})
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            messages=messages,
            tools=tools,
            max_tokens=1024
        )
        
        last_assistant_message = response.content
        print("Assistant response:")
        tool_uses = []
        
        for content_block in last_assistant_message:
            if content_block.type == 'text':
                print(content_block.text)
            elif content_block.type == 'tool_use':
                tool_uses.append(content_block)
                print(f"Tool use detected: {content_block.name}")

        messages.append({
            "role": "assistant",
            "content": last_assistant_message
        })

        if tool_uses:
            tool_results = []
            for tool_use in tool_uses:
                if tool_use.name == "execute_code_in_notebook":
                    code = tool_use.input.get('code', '')
                    results = execute_code_in_notebook(code)
                    content = "\n".join(str(r) for r in results)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": content
                    })
            
            if tool_results:
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
                
                follow_up = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=1024
                )
                
                print("\nFollow-up response:")
                for block in follow_up.content:
                    if block.type == 'text':
                        print(block.text)
                
                messages.append({
                    "role": "assistant", 
                    "content": follow_up.content
                })
            
    except KeyboardInterrupt:
        break

shutdown_kernel()
