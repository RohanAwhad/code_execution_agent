import atexit
from pypdf import PdfReader
import dataclasses
import streamlit as st
import os
import copy
import json
import base64
from io import BytesIO
from PIL import Image
from typing import Any, List, Dict
from jupyter_client.manager import KernelManager
import time

from .llm import Message 


def load_global_messages_from_disk() -> Dict[str, List[Message]]:
  if os.path.exists("./history/global_messages.json"):
    with open("./history/global_messages.json", "r") as file:
      data = json.load(file)
      return {k: [Message(**msg) for msg in v] for k, v in data.items()}
  else:
    with open("./history/global_messages.json", "w") as file:
      json.dump({}, file)
    return {}


def save_global_messages_to_disk(global_messages: Dict[str, List[Message]]) -> None:
  with open("./history/global_messages.json", "w") as file:
    json.dump({k: [dataclasses.asdict(msg) for msg in v] for k, v in global_messages.items()}, file)



def handle_image_file(file_bytes: bytes, file_extension: str) -> tuple[list[dict[str, Any]], Image.Image]:
  image_data = base64.b64encode(file_bytes).decode('utf-8')
  thumbnail = Image.open(BytesIO(file_bytes))
  message_content = [{'type': 'image_url', 'image_url': {
    'url': f'data:image/{file_extension[1:]};base64,{image_data}'
  }}]
  return message_content, thumbnail

def handle_pdf_file(file: BytesIO) -> str:
  pdf_reader = PdfReader(file)
  text_content = ""
  for page in pdf_reader.pages:
    text_content += page.extract_text() + "\n"
  return text_content

def save_file_to_data_dir(file_name: str, file_buffer: BytesIO) -> str:
  file_path = f"./data/{file_name}"
  with open(file_path, "wb") as f:
    f.write(file_buffer.getbuffer())
  return f"Stored {file_name} in the ./data directory."

def handle_text_file(file: BytesIO) -> str:
  return file.getvalue().decode('utf-8')

def process_uploaded_file(uploaded_file: BytesIO, file_name: str) -> tuple[Any, Image.Image | None]:
  file_extension = os.path.splitext(file_name)[1].lower()
  
  if file_extension in ['.png', '.jpg', '.jpeg']:
    return handle_image_file(uploaded_file.read(), file_extension)
    
  elif file_extension == '.pdf':
    return handle_pdf_file(uploaded_file), None
    
  elif file_extension in ['.txt']:
    return handle_text_file(uploaded_file), None
    
  else:
    return save_file_to_data_dir(file_name, uploaded_file), None
