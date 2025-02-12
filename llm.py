from openai import OpenAI, AsyncClient
from tqdm import tqdm
from typing import List, Union
from utils import *
import os
from openai import AzureOpenAI, AsyncAzureOpenAI
# class LLM():
#     def __init__(self, model_name, base_url, api_key, log_path):
#         self.model_name = model_name
#         self.base_url = base_url
#         self.api_key = api_key
#         self.client = OpenAI(
#             api_key=self.api_key,
#             base_url=self.base_url,
#             timeout=1200
#         )
#         self.log_path = log_path
#         self.has_log = False
#     def get_response(self, prompt:Union[list, str] , max_tokens:int, is_stream:bool = False, temperature = None, log = True,  log_message = None)->str:
#         if log and self.has_log is False:
#             try:
#                 with open(self.log_path, "w") as f:
#                     pass
#                 self.has_log = True
#             except Exception as e:
#                 print(e)
#                 print("Failed to open log file")

#         if is_stream:
#             result = ""
#             if type(prompt) == str:
#                 response = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role" : "user", "content" : prompt}
#                     ],
#                     max_tokens=max_tokens,
#                     stream=True,
#                     temperature=temperature         
#                 )
#             else:
#                 response = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=prompt,
#                     max_tokens=max_tokens,
#                     stream=True,
#                     temperature=temperature,  
#                 )
            
#             for response_item in response:
#                 if len(response_item.choices) == 0:
#                     continue
#                 message = response_item.choices[0].delta.content
#                 if message is not None:
#                     result += message
            
#         else:
#             if type(prompt) == list:
#                 response = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=prompt,
#                     max_tokens=max_tokens,
#                     temperature=temperature             
#                 )
#                 if len(response.choices) == 0:
#                     result = ""
#                 else:
#                     result = response.choices[0].message.content
#             else:
#                 response = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role" : "user", "content" : prompt}
#                     ],
#                     max_tokens=max_tokens,
#                     temperature=temperature             
#                 )
#                 if len(response.choices) == 0:
#                     result = ""
#                 else:
#                     result = response.choices[0].message.content
#         if log:
#             with open(self.log_path, "a") as f:
#                 f.write("="*50 + log_message + "="*50 + "\n") if log_message is not None else f.write("="*50 + "" + "="*50 + "\n")
#                 f.write("Prompt: \n" + str(prompt) + "\n")
#                 f.write("Response: \n" + result + "\n")
#         return result
        
# class AsyncLLM(LLM):
#     def __init__(self, model_name, base_url, api_key, log_path):
#         super().__init__(model_name, base_url, api_key, log_path)
#         self.async_client = AsyncClient(
#             api_key=self.api_key,
#             base_url=self.base_url,
#             timeout=1200,
#             max_retries = 5
#         )

#     async def get_response(self, prompt:Union[list, str], max_tokens:int, is_stream:bool = False, temperature = None, log = True,  log_message = None)->str:
#         if log and self.has_log is False:
#             try:
#                 with open(self.log_path, "w") as f:
#                     pass
#                 self.has_log = True
#             except Exception as e:
#                 print(e)
#                 print("Failed to open log file")
#         if is_stream:
#             result = ""
#             if type(prompt) == str:
#                 response = await self.async_client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role" : "user", "content" : prompt}
#                     ],
#                     max_tokens=max_tokens,
#                     stream=True,
#                     temperature=temperature
#                 )
#             else:
#                 response = await self.async_client.chat.completions.create(
#                     model=self.model_name,
#                     messages=prompt,
#                     max_tokens=max_tokens,
#                     stream=True,
#                     temperature=temperature,
#                 )
#             async for response_item in response:
#                 if len(response_item.choices) == 0:
#                     continue
#                 message = response_item.choices[0].delta.content
#                 if message is not None:
#                     result += message
#         else:
#             if type(prompt) == list:
#                 response = await self.async_client.chat.completions.create(
#                     model=self.model_name,
#                     messages=prompt,
#                     max_tokens=max_tokens,
#                     temperature=temperature
#                 )
#                 if len(response.choices) == 0:
#                     result = ""
#                 else:
#                     result = response.choices[0].message.content
#             else:
#                 response = await self.async_client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role" : "user", "content" : prompt}
#                     ],
#                     max_tokens=max_tokens,
#                     temperature=temperature
#                 )
#                 if len(response.choices) == 0:
#                     result = ""
#                 else:
#                     result = response.choices[0].message.content
#         if log:
#             with open(self.log_path, "a") as f:
#                 f.write("="*50 + log_message + "="*50 + "\n") if log_message is not None else f.write("="*50 + "" + "="*50 + "\n")
#                 f.write("Prompt: \n" + str(prompt) + "\n")
#                 f.write("Response: \n" + result + "\n")
#         return result
#     async def get_batch_response(self, prompts:List[Union[list, str]], max_tokens:int, is_stream:bool = False, temperature = None, log = True,  log_message = None)->List[str]:
#         results = await asyncio.gather(*[self.get_response(prompt, max_tokens, is_stream, temperature, log, log_message) for prompt in prompts])
#         return results
    
#     def get_response_sync(self, prompt:Union[list, str], max_tokens:int, is_stream:bool = False, temperature = None, log = True,  log_message = None)->str:
#         return super().get_response(prompt, max_tokens, is_stream, temperature, log, log_message)
    
class LLM():
    def __init__(self, args):
        """
        args need to specify some parameters that always changed such as log_path, work_dir, output_path, etc.
        """
        self.openai_config = get_config("client_config/openai_config")
        self.azure_config = get_config("client_config/azure_config")
        self.sync_client = OpenAI(
            **self.openai_config
            ) if not get_config()['enable_azure'] else AzureOpenAI(
            **self.azure_config
            )
        self.async_client = AsyncClient(
            **self.openai_config
            ) if not get_config()['enable_azure'] else AsyncAzureOpenAI(
            **self.azure_config
            )
        self.log_path = args.log_path # has log_path, if log_path is not None, then create a log file, else do nothing
        self.model_config = get_config("model_config")
        self.has_created_log = False
    @retry_on_failure_sync(max_retries=get_config()['max_retries'])
    def get_response_sync(self, prompt:Union[list, str], log_stage:Union[str, None] = None, **llm_generate_kwargs):
        model_config = union_dicts(self.model_config, llm_generate_kwargs)
        if self.has_created_log is False:
            if self.log_path is not None:
                with open(self.log_path, "w", encoding="utf-8") as f:
                    pass
                self.has_created_log = True
            else:
                pass
        prompt = prompt if type(prompt) == list else [
            {"role" : "user", "content" : prompt}
        ]
        response = self.sync_client.chat.completions.create(
            messages=prompt,
            **model_config
        )
        if model_config["stream"]:
            result = ""
            for response_item in response:
                if len(response_item.choices) == 0:
                    continue
                message = response_item.choices[0].delta.content
                if message is not None:
                    result += message
        else:
            if len(response.choices) == 0:
                result = ""
            else:
                result = response.choices[0].message.content
        log_with_stage(log_stage, prompt, result, self.log_path)
        return result
    @retry_on_failure_async(max_retries=get_config()['max_retries'], max_concurrent=get_config('async_config')['max_concurrent'])
    async def get_response_async(self, prompt:Union[list, str], log_stage: Union[str, None] = None, **llm_generate_kwargs):
        model_config = union_dicts(self.model_config, llm_generate_kwargs)
        if self.has_created_log is False:
            if self.log_path is not None:
                with open(self.log_path, "w", encoding="utf-8") as f:
                    pass
                self.has_created_log = True
            else:
                pass
        prompt = prompt if type(prompt) == list else [
            {"role" : "user", "content" : prompt}
        ]
        response = await self.async_client.chat.completions.create(
            messages=prompt,
            **model_config
        )
        if model_config["stream"]:
            result = ""
            async for response_item in response:
                if len(response_item.choices) == 0:
                    continue
                message = response_item.choices[0].delta.content
                if message is not None:
                    result += message
        else:
            if len(response.choices) == 0:
                result = ""
            else:
                result = response.choices[0].message.content
        log_with_stage(log_stage, prompt, result, self.log_path)
        return result
        
        



        
        



        


    






# max_retry_times = 5
# configs = {
#     "gpt-4o": {
#         "azure_endpoint":"",
#         "model":"gpt-4o",
#         "api_version":""
#     }
# }


# def get_client(config_name, async_mode = False):
#     config = configs[config_name]
#     if async_mode:
#         client = AsyncAzureOpenAI(
#             azure_endpoint=config["azure_endpoint"],
#             max_retries=max_retry_times,
#             api_version=config["api_version"],
#             api_key=os.environ["OPENAI_API_KEY"]
#         )
#     else:
#         client = AzureOpenAI(
#             azure_endpoint=config["azure_endpoint"],
#             max_retries=max_retry_times,
#             api_version=config["api_version"],
#             api_key=os.environ["OPENAI_API_KEY"]
#         )
#     return client


# class AzureLLM():
#     def __init__(self, config_name, log_path):
#         self.client = get_client(config_name, async_mode = True)
#         self.sync_client = get_client(config_name, async_mode = False)
#         self.log_path = log_path
#         self.has_log = False
#         self.config_name = config_name
#         self.model_name = configs[config_name]["model"]
#     async def get_response(self, prompt:Union[list, str], max_tokens:int, is_stream:bool = False, temperature = 0, log = True,  log_message = None)->str:
#         if log and self.has_log is False:
#             try:
#                 with open(self.log_path, "w") as f:
#                     pass
#                 self.has_log = True
#             except Exception as e:
#                 print(e)
#                 print("Failed to open log file")

#         if is_stream:
#             result = ""
#             if type(prompt) == str:
#                 response = await self.client.chat.completions.create(
#                     model=configs[self.config_name]["model"],
#                     messages=[
#                         {"role" : "user", "content" : prompt}
#                     ],
#                     max_tokens=max_tokens,
#                     stream=True,
#                     temperature=temperature
#                 )
#             else:
#                 response = await self.client.chat.completions.create(
#                     model=configs[self.config_name]["model"],
#                     messages=prompt,
#                     max_tokens=max_tokens,
#                     stream=True,
#                     temperature=temperature,
#                 )
#             async for response_item in response:
#                 if len(response_item.choices) == 0:
#                     continue
#                 message = response_item.choices[0].delta.content
#                 if message is not None:
#                     result += message
#         else:
#             if type(prompt) == str:
#                 response = await self.client.chat.completions.create(
#                     model=configs[self.config_name]["model"],
#                     messages=[
#                         {"role" : "user", "content" : prompt}
#                     ],
#                     max_tokens=max_tokens,
#                     temperature=temperature
#                 )
#             else:
#                 response = await self.client.chat.completions.create(
#                     model=configs[self.config_name]["model"],
#                     messages=prompt,
#                     max_tokens=max_tokens,
#                     temperature=temperature,
#                 )
#             if len(response.choices) == 0:
#                 result = ""
#             else:
#                 result = response.choices[0].message.content
#             if log:
#                 with open(self.log_path, "a") as f:
#                     f.write("="*50 + log_message + "="*50 + "\n") if log_message is not None else f.write("="*50 + "" + "="*50 + "\n")
#                     f.write("Prompt: \n" + str(prompt) + "\n")
#                     f.write("Response: \n" + result + "\n")
#         return result
#     async def get_batch_response(self, prompts:List[Union[list, str]], max_tokens:int, is_stream:bool = False, temperature = 0, log = True,  log_message = None)->List[str]:
#         if type(prompts) == str:
#             prompts = [prompts]
#         results = await asyncio.gather(*[self.get_response(prompt, max_tokens, is_stream, temperature, log, log_message) for prompt in prompts])
#         return results
    
#     def get_response_sync(self, prompt:Union[list, str], max_tokens:int, is_stream:bool = False, temperature = 0, log = True,  log_message = None)->str:
#         if log and self.has_log is False:
#             try:
#                 with open(self.log_path, "w") as f:
#                     pass
#                 self.has_log = True
#             except Exception as e:
#                 print(e)
#                 print("Failed to open log file")

#         if is_stream:
#             result = ""
#             if type(prompt) == str:
#                 response = self.sync_client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role" : "user", "content" : prompt}
#                     ],
#                     max_tokens=max_tokens,
#                     stream=True,
#                     temperature=temperature         
#                 )
#             else:
#                 response = self.sync_client.chat.completions.create(
#                     model=self.model_name,
#                     messages=prompt,
#                     max_tokens=max_tokens,
#                     stream=True,
#                     temperature=temperature,  
#                 )
            
#             for response_item in response:
#                 if len(response_item.choices) == 0:
#                     continue
#                 message = response_item.choices[0].delta.content
#                 if message is not None:
#                     result += message
            
#         else:
#             if type(prompt) == list:
#                 response = self.sync_client.chat.completions.create(
#                     model=self.model_name,
#                     messages=prompt,
#                     max_tokens=max_tokens,
#                     temperature=temperature             
#                 )
#                 if len(response.choices) == 0:
#                     result = ""
#                 else:
#                     result = response.choices[0].message.content
#             else:
#                 response = self.sync_client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role" : "user", "content" : prompt}
#                     ],
#                     max_tokens=max_tokens,
#                     temperature=temperature             
#                 )
#                 if len(response.choices) == 0:
#                     result = ""
#                 else:
#                     result = response.choices[0].message.content
#         if log:
#             with open(self.log_path, "a") as f:
#                 f.write("="*50 + log_message + "="*50 + "\n") if log_message is not None else f.write("="*50 + "" + "="*50 + "\n")
#                 f.write("Prompt: \n" + str(prompt) + "\n")
#                 f.write("Response: \n" + result + "\n")
#         return result

        
