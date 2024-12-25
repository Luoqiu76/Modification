import asyncio
from contextlib import contextmanager
from functools import wraps
import inspect
import json
import random
import time
from typing import Callable, List, Tuple, Union
from errors import Errors


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

def load_datas(input_path: str) -> List[dict]:
    if input_path.endswith(".json"):
        return load_json_datas(input_path)   
    else:
        return load_jsonl_datas(input_path)
    



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

def concurrency_limit_with_tracking_and_rate_limit(max_concurrent: Union[int, None], rate_limit: Union[int, None]):
    """
    Limit the maximum number of concurrent tasks of an async function and track the current and maximum number of concurrent tasks
    """
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None
    tracker = ConcurrencyTracker()  

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if rate_limit is not None:
                await asyncio.sleep(rate_limit)
            if semaphore is not None:
                async with semaphore:
                    await tracker.increment()
                    try:
                        result =  await func(*args, **kwargs)
                    finally:
                        await tracker.decrement()
            else:
                await tracker.increment()
                try:
                    result = await func(*args, **kwargs)
                finally:
                    await tracker.decrement()
            if rate_limit is not None:
                await asyncio.sleep(rate_limit)
            return result
        wrapper.tracker = tracker  
        return wrapper
    return decorator


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

async def execute_with_retry_async(func, max_retries=5, enable_wait = True, *args, **kwargs)->Tuple[Union[any, Errors], int]:
    """
    for the high level async function, need to return the time in the future itself
    """
    import traceback
    retries = 0
    errors = Errors([])
    while retries < max_retries:
        with timed_block() as timer:
            try:
                result = await func(*args, **kwargs)
                return (result, timer() / (retries + 1))
            except Exception as e:
                retries += 1
                errors.append((e, traceback.format_exc()))
                # print(f"Error: {e}")
                # print (f"Traceback: {traceback.format_exc()}")
                if enable_wait:
                    await asyncio.sleep(get_wait_time(retries))
                if retries >= max_retries:
                    return (errors, timer())          



def process_json_output(output: str):
    error = {
        "error" : "",
        "output" : None,
        "original_output" : None
    }
    orginal_output = output
    import re
    pattern = r"```"
    match = re.search(pattern, output)
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
    

    start = match.start()
    output = output[::-1]
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


def execute_with_retry(func, max_retries=5, enable_wait = True, *args, **kwargs)->Tuple[Union[any, Errors], int]:
    import traceback
    retries = 0
    errors = Errors([])
    while retries < max_retries:
        with timed_block() as timer:
            try:
                # 执行目标函数
                if inspect.iscoroutinefunction(func):
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(func(*args, **kwargs))
                    loop.close()
                    return (result, timer() / (retries + 1))
                else:
                    return (func(*args, **kwargs), timer() / (retries + 1))
            except Exception as e:
                retries += 1
                errors.append((e, traceback.format_exc()))
                # print(f"Error: {e}")
                # print (f"Traceback: {traceback.format_exc()}")
                if enable_wait:
                    time.sleep(get_wait_time(retries))
                if retries >= max_retries:
                    return (errors, timer())