
import logging
import os
import sys
import warnings

from google.genai import Client

logger = logging.getLogger(__name__)
warnings.warn("Model as AI agentic backend is outdated. Please use the newer OpenRouter adaptation")

client = Client(api_key=os.getenv("GEMINI_API_KEY"))
print(f"Passed arguments are: {sys.argv}")
delete_files = False
if len(sys.argv) > 1:
      delete_files = eval(sys.argv[1])
      if type(delete_files) != bool:
            raise ValueError("First argument should be boolean")
print(f"Delete files takes on: {delete_files}")

print("="*30)
for file in client.files.list():
      print(f"FILE: {file}")
      if delete_files:
            client.files.delete(name=file.name)
      print("\n\n")