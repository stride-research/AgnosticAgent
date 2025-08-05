import yaml
import argparse

FILE_PATH = "examples/config.yaml"

with open(file=FILE_PATH) as file:
      CONFIG_DICT = yaml.safe_load(file.read())

print(f"CONFIG DICT IS: {CONFIG_DICT}")

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=CONFIG_DICT["DEFAULT_BACKEND"], choices=["ollama", "openrouter", "openai"])
parser.add_argument("--model", default=CONFIG_DICT["DEFAULT_MODEL"])
inline_args = parser.parse_args()