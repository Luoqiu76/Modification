from llm import LLM
from utils import *
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--log_path", type=str, default = None)
args = parser.parse_args()
llm = LLM(
    args = args
)
llm_kwargs = get_config("model_config")
response = llm.get_response_sync(
    prompt = "What is your name",
    **llm_kwargs
)
print(response)

