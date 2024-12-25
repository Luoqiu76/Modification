from openai import OpenAI, AsyncClient
from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union
import asyncio
import os

class LLM():
    def __init__(self, model_name, base_url, api_key, log_path):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=1200
        )
        self.log_path = log_path
        self.has_log = False
    def get_response(self, prompt:Union[list, str] , max_tokens:int, is_stream:bool = False, temperature = None, log = True,  log_message = None)->str:
        if log and self.has_log is False:
            try:
                with open(self.log_path, "w") as f:
                    pass
                self.has_log = True
            except Exception as e:
                print(e)
                print("Failed to open log file")

        if is_stream:
            result = ""
            if type(prompt) == str:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role" : "user", "content" : prompt}
                    ],
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=temperature         
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=temperature,  
                )
            
            for response_item in response:
                if len(response_item.choices) == 0:
                    continue
                message = response_item.choices[0].delta.content
                if message is not None:
                    result += message
            
        else:
            if type(prompt) == list:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature             
                )
                if len(response.choices) == 0:
                    result = ""
                else:
                    result = response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role" : "user", "content" : prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature             
                )
                if len(response.choices) == 0:
                    result = ""
                else:
                    result = response.choices[0].message.content
        if log:
            with open(self.log_path, "a") as f:
                f.write("="*50 + log_message + "="*50 + "\n") if log_message is not None else f.write("="*50 + "" + "="*50 + "\n")
                f.write("Prompt: \n" + str(prompt) + "\n")
                f.write("Response: \n" + result + "\n")
        return result
        
class AsyncLLM(LLM):
    def __init__(self, model_name, base_url, api_key, log_path):
        super().__init__(model_name, base_url, api_key, log_path)
        self.async_client = AsyncClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=1200,
            max_retries = 5
        )

    async def get_response(self, prompt:Union[list, str], max_tokens:int, is_stream:bool = False, temperature = None, log = True,  log_message = None)->str:
        if log and self.has_log is False:
            try:
                with open(self.log_path, "w") as f:
                    pass
                self.has_log = True
            except Exception as e:
                print(e)
                print("Failed to open log file")
        if is_stream:
            result = ""
            if type(prompt) == str:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role" : "user", "content" : prompt}
                    ],
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=temperature
                )
            else:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=temperature,
                )
            async for response_item in response:
                if len(response_item.choices) == 0:
                    continue
                message = response_item.choices[0].delta.content
                if message is not None:
                    result += message
        else:
            if type(prompt) == list:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                if len(response.choices) == 0:
                    result = ""
                else:
                    result = response.choices[0].message.content
            else:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role" : "user", "content" : prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                if len(response.choices) == 0:
                    result = ""
                else:
                    result = response.choices[0].message.content
        if log:
            with open(self.log_path, "a") as f:
                f.write("="*50 + log_message + "="*50 + "\n") if log_message is not None else f.write("="*50 + "" + "="*50 + "\n")
                f.write("Prompt: \n" + str(prompt) + "\n")
                f.write("Response: \n" + result + "\n")
        return result
    async def get_batch_response(self, prompts:List[Union[list, str]], max_tokens:int, is_stream:bool = False, temperature = None, log = True,  log_message = None)->List[str]:
        results = await asyncio.gather(*[self.get_response(prompt, max_tokens, is_stream, temperature, log, log_message) for prompt in prompts])
        return results
    
    def get_response_sync(self, prompt:Union[list, str], max_tokens:int, is_stream:bool = False, temperature = None, log = True,  log_message = None)->str:
        return super().get_response(prompt, max_tokens, is_stream, temperature, log, log_message)
    

class TextSplitter():
    def __init__(self, model_name, base_url, api_key, breakpoint_threshold_type ,breakpoint_threshold_amount):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.text_splitter = SemanticChunker(OpenAIEmbeddings(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            check_embedding_ctx_length=False,
            chunk_size=6
        ),breakpoint_threshold_type=self.breakpoint_threshold_type,
        sentence_split_regex=r"[。？！\n]",
        breakpoint_threshold_amount = breakpoint_threshold_amount)
    def get_chunks(self, docs : List[str]) -> List[str]:
        docs = self.text_splitter.create_documents(docs)
        return [docs[i].page_content for i in range(len(docs))]
    
class FixedTokensTextSplitter():
    def __init__(self, **kwargs):
        self.text_splitter = RecursiveCharacterTextSplitter(
            **kwargs
        )

    def get_chunks(self, docs : Union[List[str], str]) -> List[str]:
        if type(docs) == str:
            docs = [docs]
        docs = self.text_splitter.create_documents(docs)
        return [docs[i].page_content for i in range(len(docs))]
    



from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI, AsyncAzureOpenAI


max_retry_times = 5
configs = {
    "gpt-4o": {
        "azure_endpoint":"",
        "model":"gpt-4o",
        "api_version":""
    }
}


def get_client(config_name, async_mode = False):
    config = configs[config_name]
    if async_mode:
        client = AsyncAzureOpenAI(
            azure_endpoint=config["azure_endpoint"],
            max_retries=max_retry_times,
            api_version=config["api_version"],
            api_key=os.environ["OPENAI_API_KEY"]
        )
    else:
        client = AzureOpenAI(
            azure_endpoint=config["azure_endpoint"],
            max_retries=max_retry_times,
            api_version=config["api_version"],
            api_key=os.environ["OPENAI_API_KEY"]
        )
    return client


class AzureLLM():
    def __init__(self, config_name, log_path):
        self.client = get_client(config_name, async_mode = True)
        self.sync_client = get_client(config_name, async_mode = False)
        self.log_path = log_path
        self.has_log = False
        self.config_name = config_name
        self.model_name = configs[config_name]["model"]
    async def get_response(self, prompt:Union[list, str], max_tokens:int, is_stream:bool = False, temperature = 0, log = True,  log_message = None)->str:
        if log and self.has_log is False:
            try:
                with open(self.log_path, "w") as f:
                    pass
                self.has_log = True
            except Exception as e:
                print(e)
                print("Failed to open log file")

        if is_stream:
            result = ""
            if type(prompt) == str:
                response = await self.client.chat.completions.create(
                    model=configs[self.config_name]["model"],
                    messages=[
                        {"role" : "user", "content" : prompt}
                    ],
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=temperature
                )
            else:
                response = await self.client.chat.completions.create(
                    model=configs[self.config_name]["model"],
                    messages=prompt,
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=temperature,
                )
            async for response_item in response:
                if len(response_item.choices) == 0:
                    continue
                message = response_item.choices[0].delta.content
                if message is not None:
                    result += message
        else:
            if type(prompt) == str:
                response = await self.client.chat.completions.create(
                    model=configs[self.config_name]["model"],
                    messages=[
                        {"role" : "user", "content" : prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                response = await self.client.chat.completions.create(
                    model=configs[self.config_name]["model"],
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            if len(response.choices) == 0:
                result = ""
            else:
                result = response.choices[0].message.content
            if log:
                with open(self.log_path, "a") as f:
                    f.write("="*50 + log_message + "="*50 + "\n") if log_message is not None else f.write("="*50 + "" + "="*50 + "\n")
                    f.write("Prompt: \n" + str(prompt) + "\n")
                    f.write("Response: \n" + result + "\n")
        return result
    async def get_batch_response(self, prompts:List[Union[list, str]], max_tokens:int, is_stream:bool = False, temperature = 0, log = True,  log_message = None)->List[str]:
        if type(prompts) == str:
            prompts = [prompts]
        results = await asyncio.gather(*[self.get_response(prompt, max_tokens, is_stream, temperature, log, log_message) for prompt in prompts])
        return results
    
    def get_response_sync(self, prompt:Union[list, str], max_tokens:int, is_stream:bool = False, temperature = 0, log = True,  log_message = None)->str:
        if log and self.has_log is False:
            try:
                with open(self.log_path, "w") as f:
                    pass
                self.has_log = True
            except Exception as e:
                print(e)
                print("Failed to open log file")

        if is_stream:
            result = ""
            if type(prompt) == str:
                response = self.sync_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role" : "user", "content" : prompt}
                    ],
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=temperature         
                )
            else:
                response = self.sync_client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=temperature,  
                )
            
            for response_item in response:
                if len(response_item.choices) == 0:
                    continue
                message = response_item.choices[0].delta.content
                if message is not None:
                    result += message
            
        else:
            if type(prompt) == list:
                response = self.sync_client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature             
                )
                if len(response.choices) == 0:
                    result = ""
                else:
                    result = response.choices[0].message.content
            else:
                response = self.sync_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role" : "user", "content" : prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature             
                )
                if len(response.choices) == 0:
                    result = ""
                else:
                    result = response.choices[0].message.content
        if log:
            with open(self.log_path, "a") as f:
                f.write("="*50 + log_message + "="*50 + "\n") if log_message is not None else f.write("="*50 + "" + "="*50 + "\n")
                f.write("Prompt: \n" + str(prompt) + "\n")
                f.write("Response: \n" + result + "\n")
        return result
    
