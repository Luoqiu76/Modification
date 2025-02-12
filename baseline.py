import tiktoken
from llm import LLM
from utils import *
class Baseline:
    def __init__(self, llm: LLM, prompts: dict, llm_kwargs: dict):
        self.llm = llm
        self.prompts = prompts
        self.llm_kwargs = llm_kwargs
    @retry_on_failure_sync(max_retries=get_config()['max_retries'], return_time=True)
    def forward_sync(self, text: str, overall_modification: str):
        prompt = self.prompts["baseline"].format(
            text=text,
            modification=overall_modification
        )
        response = self.llm.get_response_sync(
            prompt=prompt,
            log_stage="Baseline",

            **self.llm_kwargs
        )
        return response
    @retry_on_failure_async(max_retries=get_config()['max_retries'], max_concurrent=get_config('async_config')['max_concurrent'], return_time=True)
    async def forward_async(self, text: str, overall_modification: str):

        prompt = self.prompts["baseline"].format(
            text=text,
            modification=overall_modification
        )
        max_tokens_dict = {
            "gpt-3.5-turbo": 4096 - len(tiktoken.get_encoding("cl100k_base").encode(prompt)),
            "gpt-4o" : 16384,
            "gpt-4o-mini" : 16384,
            "qwen-turbo-1101" : 8192,
            "gpt-4-32k" : 16384
        }
        self.llm_kwargs['max_tokens'] = max_tokens_dict[self.llm_kwargs['model']]
        response = await self.llm.get_response_async(
            prompt=prompt,
            log_stage="Baseline",
            **self.llm_kwargs
        )
        return response
