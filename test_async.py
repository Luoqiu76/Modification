import asyncio
import functools
import random
import time
from functools import wraps
import traceback
from typing import Callable, Any, Tuple, List
from contextlib import contextmanager

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


# 示例异步函数，可能会失败

async def unreliable_function(task_id):
    from random import randint
    await asyncio.sleep(1)  # 模拟异步操作
    result = randint(1, 10)
    if result > 9:
        print(f"Task {task_id}: Success!")
        return result
    else:
        print(f"Task {task_id}: Failure! Result was {result}")
        raise ValueError("Random value too low")


def unreliable_function_sync(task_id):
    from random import randint
    time.sleep(1)
    result = randint(1, 10)
    if result > 9:
        print(f"Task {task_id}: Success!" )
        return result
    else:
        print(f"Task {task_id}: Failure!" )
        raise ValueError("Random value too low")
    

# 使用装饰器的异步函数调用

async def main_async():
    tasks = [unreliable_function(task_id) for task_id in range(10)]
    results = await asyncio.gather(*tasks)
    return results


# 使用装饰器的同步函数调用

def main_sync():
    tasks = [unreliable_function_sync(task_id) for task_id in range(10)]
    results = tasks
    return results

# 计算二者执行时间


main_sync = retry_on_failure_sync(max_retries=0, return_time=True)(main_sync)
main_async = retry_on_failure_async(max_retries=0, max_concurrent=1, return_time=True)(main_async)
unreliable_function_sync = retry_on_failure_sync(max_retries=5, return_time=True)(unreliable_function_sync)
unreliable_function = retry_on_failure_async(max_retries=5, max_concurrent=1, return_time=True)(unreliable_function)

if __name__ == '__main__':
    
    async def run():
        result, async_time = await main_async()
        print(f"Async function took {async_time:.2f} seconds on average")
        print(result)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    result, sync_time = main_sync()
    print(f"Sync function took {sync_time:.2f} seconds on average")
    print(result)
        
