from jupyter_client.manager import KernelManager
from typing import Any


def execute_code_in_notebook(code: str, kernel_manager: KernelManager, kernel_client) -> list[Any]:
  if not code:
    return []
  print('Code:')
  print(code)
  kernel_client.execute(code)
  output_content: str = ""
  outputs: list[Any] = []
  iter = 0
  while True:
    iter += 1
    try:
      msg: dict[str, Any] = kernel_client.get_iopub_msg()
      print(msg)
      print(iter)
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
        print('Kernel client shut down.')
    print('Shutting Manager down ... ')
    if kernel_manager is not None:
        kernel_manager.shutdown_kernel()
        kernel_manager.cleanup_resources()
        print('Kernel manager shut down.')


if __name__ == '__main__':
  kernel_manager = KernelManager()
  kernel_manager.start_kernel()
  kernel_client = kernel_manager.client()
  kernel_client.start_channels()
  kernel_client.wait_for_ready()

  #code = 'print("hello")'
  code = 's = "hello";print(s); import time; time.sleep(5); print(2)'
  execute_code_in_notebook(code, kernel_manager, kernel_client)
  shutdown_kernel(kernel_manager, kernel_client)
