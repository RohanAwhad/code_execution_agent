from typing import Any


def execute_code_in_notebook(code: str, kernel_client) -> list[Any]:
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
      msg: dict[str, Any] = kernel_client.get_iopub_msg()
      if msg['msg_type'] == 'execute_result':
        outputs.append(msg['content']['data']['text/plain'])
      elif msg['msg_type'] == 'display_data':
        if 'image/png' in msg['content']['data']:
          outputs.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + msg['content']['data']['image/png']}})
      elif msg['msg_type'] == 'stream':
        output_content += msg['content']['text']
      elif msg['msg_type'] == 'error':
        outputs.append("\n".join(msg['content']['traceback']))
      if msg['msg_type'] == 'status':
        if msg['content']['execution_state'] == 'idle':
          break
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
