import asyncio
from contextlib import contextmanager
from functools import wraps
import functools
import inspect
import json
import random
import time
import traceback
from typing import Callable, List, Tuple, Union
from errors import Errors
from openai import OpenAI, AsyncClient
from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Union
from utils import *
import asyncio
import os

class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Errors):
            return str(o)
        return super().default(o)


def load_datas(input_path: str) -> List[dict]:
    def load_jsonl_datas(input_path) -> List[dict]:
        datas = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                datas.append(data)
        return datas

    def load_json_datas(input_path: str) -> List[dict]:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    if input_path.endswith(".json"):
        return load_json_datas(input_path)   
    else:
        return load_jsonl_datas(input_path)

def process_config(config: dict) -> dict:

    if config["api_key"] =="NOT_GIVEN":
        config["api_key"] = os.getenv("OPENAI_API_KEY")

def get_config(config_name : Union[str, None] = None) -> dict:
    """
    Get the config from the config.json file.
    If config_name is spilted by "/", get the lower level config.
    eg. A/B then get obj["A"]["B"]
    """
    if hasattr(get_config, "config"):
        config = getattr(get_config, "config")
    else:
        with open(r"./config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        config['client_config/azure_config'] = process_config(config['client_config/azure_config'])
        setattr(get_config, "config", config)

    if config_name is None:
        return config
    if "/" in config_name:
        keys = config_name.split("/")
        for key in keys:
            config = config[key]
        return config
    else:
        return config[config_name]

def union_dicts(dict_a: dict, dict_b: dict) -> dict:
    """
    Merge two dictionaries into a new dictionary. If there are duplicate keys, the value in dict_b will be used.
    """
    merged_dict = {}
    for key in dict_a:
        if key in dict_b:
            merged_dict[key] = dict_b[key]
        else:
            merged_dict[key] = dict_a[key]
    for key in dict_b:
        if key not in dict_a:
            merged_dict[key] = dict_b[key]
    return merged_dict

def log_with_stage(stage: Union[str, None], prompt : str, response: Union[str, List[str]], log_path: Union[str, None] = None):
    log_stage = "" if stage is None else stage
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
                try:
                    f.write("="*50 + log_stage + "="*50 + "\n")
                    f.write("Prompt: \n" + str(prompt) + "\n")
                    f.write("Response: \n" + str(response) + "\n")
                    f.write("="*100 + "\n")
                except Exception as e:
                    raise e




 



def process_json_output(output: str) -> dict:
    error = {
        "error" : "",
        "output" : None,
        "original_output" : None
    }
    orginal_output = output
    import re
    pattern = r"```"
    match = re.search(pattern, output)
    # find the start of the JSON output
    if match is None:
        pattern = r"\["
        match = re.search(pattern, output)
        if match is None:
            pattern = r"\{"
            match = re.search(pattern, output)
            if match is None:
                error["error"] = "No JSON output start found in the response."
                error["output"] = output
                error['original_output'] = orginal_output
                raise Exception(error)
            else:
                pass
        else:
            pass
    else:
        pass
    

    start = match.start()
    output = output[::-1]
    # find the end of the JSON output
    if pattern == r"\[":
        pattern = r"\]"
    elif pattern == r"\{":
        pattern = r"\}"
    else:
        pass
    match = re.search(pattern, output)
    if match is None:
        error["error"] = "No JSON output end found in the response."
        error["output"] = output
        error['original_output'] = orginal_output
        raise Exception(error)
    end = match.start()
    end = len(output) - end
    output = output[::-1]
    output = output[start:end]
    output = output.strip()
    if output.startswith("```json") and output.endswith("```"):
        output = output[7:-3]
    elif output.startswith("```") and output.endswith("```"):
        output = output[3:-3]
    elif (output.startswith("[") and output.endswith("]")) or (output.startswith("{") and output.endswith("}")):
        pass
    try:
        output = json.loads(output)
    except Exception as e:
        error["output"] = output
        error['original_output'] = orginal_output
        error['error'] = str(e)
        raise Exception(error)
    return output

                


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
    





class ConcurrencyTracker:
    def __init__(self):
        self.current_concurrent = 0  
        self.max_concurrent_seen = 0  
        self.lock = asyncio.Lock()  

    async def increment(self):
        async with self.lock:  
            self.current_concurrent += 1
            if self.current_concurrent > self.max_concurrent_seen:
                self.max_concurrent_seen = self.current_concurrent

    async def decrement(self):
        async with self.lock:
            self.current_concurrent -= 1



def get_wait_time(attempt: int, base_delay: float = 1.0, backoff_factor: float = 1.2, jitter_factor: float = 0.2) -> float:
    """
    :param attempt: retry attempt number, from 0
    :param base_delay: initial delay in seconds, defaults to 1.0.
    :param backoff_factor: delay factor, defaults to 2.0.
    :param jitter_factor: max jitter factor, defaults to 0.5. The jitter factor is used to add randomness to the delay. at most +/- jitter_factor * delay.
    :return: actual delay in seconds
    """
    delay = base_delay * (backoff_factor ** attempt)
    jitter = (random.uniform(-1, 1) * jitter_factor * delay)
    actual_delay = max(0, delay + jitter)
    return actual_delay

def concurrency_limit_with_tracking(max_concurrent: int):
    """
    Limit the maximum number of concurrent tasks of an async function and track the current and maximum number of concurrent tasks.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tracker = ConcurrencyTracker()

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await tracker.increment()
            async with semaphore:
                try:
                    result = await func(*args, **kwargs)
                finally:
                    await tracker.decrement()
            return result
        wrapper.tracker = tracker  
        return wrapper
    return decorator

@contextmanager
def timed_block():
    """
    with timed_block("My operation"):
        time.sleep(2)
    """
    start_time = time.time()
    try:
        yield lambda : time.time() - start_time
    finally:
        pass

class Errors:
    def __init__(self, error_and_traceback: List[Tuple[Exception, str]]):
        self.error_and_traceback = error_and_traceback

    def __str__(self):
        result = ""
        for i, (error, traceback) in enumerate(self.error_and_traceback):
            result += f"Try {i+1}: \n"
            result += f"Error : {error}\n"
            result += f"Traceback : {traceback}\n"
        return result

    def __dict__(self):
        return [
            {
                "try": i,
                "error": str(error),
                "traceback": str(traceback)
            } for i, (error, traceback) in enumerate(self.error_and_traceback)
        ]

    def __iter__(self):
        return iter(self.error_and_traceback)

    def __len__(self):
        return len(self.error_and_traceback)

    def append(self, error_and_traceback: Tuple[Exception, str]):
        self.error_and_traceback.append(error_and_traceback)

def retry_on_failure_async(max_retries=3, max_concurrent=5, return_time=False):
    """
    装饰器：如果被装饰的异步函数抛出异常，则会重试指定次数，并限制并发任务数量。
    
    :param max_retries: 最大重试次数，默认为3次。
    :param max_concurrent: 并发任务的最大数量，默认为5个。
    :param return_time: 是否返回执行时间，默认为False。
    """
    concurrency_decorator = concurrency_limit_with_tracking(max_concurrent)

    def decorator(func):
        @concurrency_decorator
        @wraps(func)
        async def wrapper(*args, **kwargs):
            errors = []
            retries = 0
            with timed_block() as timer:
                while retries <= max_retries:
                    try:
                        result = await func(*args, **kwargs)
                        if return_time:
                            elapsed_time = timer()
                            return result, elapsed_time / (retries + 1)
                        else:
                            return result
                    except Exception as e:
                        retries += 1
                        error_info = (e, repr(traceback.format_exc()))
                        errors.append(error_info)
                        elapsed_time = timer()
                        if retries > max_retries:
                            return Errors(errors), elapsed_time / (retries + 1) if return_time else Errors(errors)
                        else:
                            await asyncio.sleep(get_wait_time(retries))  # 等待一段时间后重试
        return wrapper
    return decorator

def retry_on_failure_sync(max_retries=3, return_time=False):
    """
    装饰器：如果被装饰的函数抛出异常，则会重试指定次数。
    
    :param max_retries: 最大重试次数，默认为3次。
    :param return_time: 是否返回执行时间，默认为False。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            errors = []
            with timed_block() as timer:
                while retries <= max_retries:
                    try:
                        result  = func(*args, **kwargs)
                        if return_time:
                            elapsed_time = timer()
                            return result, elapsed_time / (retries + 1)
                        else:
                            return result
                    except Exception as e:
                        retries += 1
                        error_info = (e, repr(traceback.format_exc()))
                        errors.append(error_info)
                        elapsed_time = timer()
                        if retries > max_retries:
                            return Errors(errors), elapsed_time / (retries + 1) if return_time else Errors(errors)
                        else:
                            delay_seconds = get_wait_time(retries)
                            time.sleep(delay_seconds)
        return wrapper
    return decorator