from contextlib import contextmanager
from functools import wraps
import math
from prompt import *
from llm import *
import json
import os
from typing import Callable, List, Tuple,TypedDict, Coroutine, overload
from argparse import ArgumentParser
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import tiktoken
import time
from errors import Errors 
from utils import *



def get_prompt(language : str):
    import prompt
    prompt_name_list = []
    for prompt_name in dir(prompt):
        if prompt_name.endswith("_ZH") or prompt_name.endswith("_EN"):
            prompt_name_list.append(prompt_name[:-3])
    prompt_name_list = list(set(prompt_name_list))
    if language.lower() == "en":
        for prompt_name in prompt_name_list:
            globals()[prompt_name] = getattr(prompt, prompt_name + "_EN")
    else:
        for prompt_name in prompt_name_list:
            globals()[prompt_name] = getattr(prompt, prompt_name + "_ZH")
        






class Node:
    def __init__(self, text, children, parent, start, end, summary = ""):
        self.text = text
        self.children = children
        self.parent = parent
        self.start = start
        self.end = end
        self.summary = summary
    def __str__(self):
        return str(
            {
                "from" : self.start,
                "to" : self.end,
                "summary" : self.summary,
            }
        )
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)


class Entity(TypedDict):
    entity : str
    modification : str
    importance : int

@overload
def get_entitys(text:str, llm:AsyncLLM, **kwargs) -> List[Entity]:
    ...

def get_entitys(text:str, llm:Union[LLM, AsyncLLM], **kwargs) -> List[Entity]:
    prompt = EXTRACT_ENTITY_PROMPT.format( # type: ignore
        content = text
    )   
    response = llm.get_response(prompt, log_message="Extracting entities", **kwargs) if type(llm) is LLM else llm.get_response_sync(prompt, log_message="Extracting entities", **kwargs)
    entities = process_json_output(response)
    return entities


async def baseline_async(args, text, overall_modification:str):
    if not args.azure:
        llm = AsyncLLM(
            model_name=args.chat_model,
            base_url=args.chat_base_url,
            api_key=args.chat_api_key,
            log_path=args.log_path
        )
    else:
        llm = AzureLLM(
            config_name=os.environ["CONFIG_NAME"],
            log_path=args.log_path
        )
    baseline_prompt = BASELINE_PROMPT.format( # type: ignore
        text = text,
        modification = overall_modification
    )
    max_tokens_dict = {
        "gpt-3.5-turbo": 4096 - len(tiktoken.get_encoding("cl100k_base").encode(baseline_prompt)),
        "gpt-4o" : 16384,
        "gpt-4o-mini" : 16384,
        "qwen-turbo-1101" : 8192
    }
    llm_args = {
            "max_tokens" : max_tokens_dict[args.chat_model],
            "is_stream" : args.enable_stream,
            "temperature" : 0,
            "log" : False
    }
    response = await llm.get_response(baseline_prompt, **llm_args)

    return response

def baseline_sync(args, text, overall_modification:str):
    llm = LLM(
        model_name=args.chat_model,
        base_url=args.chat_base_url,
        api_key=args.chat_api_key,
        log_path=args.log_path
    )
    baseline_prompt = BASELINE_PROMPT.format( # type: ignore
        text = text,
        modification = overall_modification
    )
    max_tokens_dict = {
        "gpt-3.5-turbo": 4096 - len(tiktoken.get_encoding("cl100k_base").encode(baseline_prompt)),
        "gpt-4o" : 16384,
        "gpt-4o-mini" : 16384,
        "qwen-turbo-1101" : 8192
    }
    llm_args = {
            "max_tokens" : max_tokens_dict[args.chat_model],
            "is_stream" : args.enable_stream,
            "temperature" : 0,
            "log" : False
    }
    response = llm.get_response(baseline_prompt, **llm_args)
    return response
    

def find_start_end(text:str, llm:LLM, node:Node, **kwargs)->bool:
    start = node['start']
    end = node['end']
    find_prompt = FIND_PROMPT.format( # type: ignore
        text = text,
        start = start,
        end = end
    )
    response = llm.get_response(find_prompt, log_message="Finding start and end position", **kwargs)
    node['text'] = response
    len_text = len(text)
    len_response = len(response)    
    if min(len_text, len_response) / max(len_text, len_response) > 0.95:
        return True
    return False




async def get_level_nodes_async(args, text:str, father:Union[Node, None], llm:AsyncLLM, entitys:List[Entity], depth , **kwargs) -> List[Node]:
    # if depth > args.depth_limit then return []
    if depth > args.depth_limit:
        return []
    
    text = text.strip()
    prompt = GET_LEVEL_NODES_PROMPT.format( # type: ignore
        text = text,
        entities = str(entitys)
    )
    # if the response is "done" means the current node has no need to split further, return []
    response = await llm.get_response(prompt, log_message= "Getting level nodes", **kwargs)
    if response.strip().lower() == "done":
        return []
    nodes = process_json_output(response)
    nodes = [
        Node(
            text = "",
            children = [],
            parent = father,
            start = node['start'],
            end = node['end'],
            summary = node['summary']
        )
        for node in nodes
    ]
    
    find_responses = await llm.get_batch_response(prompts = [FIND_PROMPT.format( # type: ignore
        text = text,
        start = node['start'],
        end = node['end']
    ) for node in nodes], **kwargs)
    
    # if the response is too similar to the text, then return []
    have_dones = [min(len(text), len(response)) / max(len(text), len(response)) > 0.95 for  response in find_responses]
    if any(have_dones):
        return []
    
    for node, text in zip(nodes, find_responses):
        node['text'] = text
    
    childrens =await asyncio.gather(*[get_level_nodes_async(args, node['text'], node, llm, entitys, depth + 1, **kwargs) for node in nodes])
    for node, children in zip(nodes, childrens):
        node['children'] = children
    father['children'].extend(nodes)
    return nodes


def get_level_nodes_sync(args, text:str, father:Union[Node, None], llm:LLM, entitys:List[Entity], depth, **kwargs) -> List[Node]:

    if depth > args.depth_limit:
        return []
    text = text.strip()
    prompt = GET_LEVEL_NODES_PROMPT.format( # type: ignore
        text = text,
        entities = str(entitys)
    )
    response = llm.get_response(prompt, log_message= "Getting level nodes", **kwargs)
    if response.strip().lower() == "done":
        return []
    nodes = process_json_output(response)
    nodes = [
        Node(
            text = "",
            children = [],
            parent = father,
            start = node['start'],
            end = node['end'],
            summary = node['summary']
        )
        for node in nodes
    ]
    have_dones = [find_start_end(text, llm, node, **kwargs) for node in nodes]
    if any(have_dones):
        return []
    for node in nodes:
        node['father'] = father
        father['children'].append(node)
        node['children'] = get_level_nodes_sync(args, node['text'], node, llm, entitys,depth + 1, **kwargs)
    return nodes
        
        
            
# def dfs(root:Node, dot: Digraph, father:str):
#     name = "from " + root.start[:10] + " to " + root.end[:10]
#     dot.node(name, root.summary)
#     if father is not None:
#         dot.edge(father, name)
#     if root.children is not None:
#         for child in root.children:
#             dfs(child, dot, name)


# def visualize_tree(root:Node):
#     dot = Digraph()
#     dfs(root, dot, None)
#     dot.render("tree", view=True, format="pdf")

def get_tree(root:Node, path:str, store:bool = False):
    def dfs(root:Node, level):
        result = {
            "level" : level,
            "summary" : root.summary,
            "children" : [],
            "start" : root.start,
            "end" : root.end
        }
        if root.children is None:
            return result
        for child in root.children:
            child_json = dfs(child, level + 1)
            result["children"].append(child_json)
        return result
    tree_json = dfs(root, 0)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree_json, f, ensure_ascii=False, indent=4)
    return tree_json
@overload
def tree_based(overall_modification:str, tree:Node, llm:AsyncLLM, **kwargs) -> str:
    ...
def tree_based(overall_modification, tree, llm:Union[LLM, AsyncLLM], **kwargs) -> str:
    prompt = BASE_TREE_PROMPT.format(  # type: ignore
        tree = str(tree),
        modification = overall_modification
    )
    response = llm.get_response(prompt=prompt, **kwargs) if type(llm) is LLM else llm.get_response_sync(prompt=prompt, **kwargs)
    return response

async def tree_based_async(overall_modification:str, tree:Node, llm:AsyncLLM, **kwargs) -> str:
    prompt = BASE_TREE_PROMPT.format(  # type: ignore
        tree = str(tree),
        modification = overall_modification
    )
    response = await llm.get_response(prompt=prompt, **kwargs)
    return response

@overload
def final_modify(args, text:str, tree:Node, llm:AsyncLLM, **kwargs) -> str:
    ...

def final_modify(args, text, tree, llm:Union[LLM, AsyncLLM], **kwargs):
    if not args.enable_chunks:
        prompt = FINAL_MODIFY.format( # type: ignore
            text = text,
            tree = str(tree)
        )
        response = llm.get_response(prompt= prompt, **kwargs) if type(llm) is LLM else llm.get_response_sync(prompt=prompt, **kwargs)
    else:
        text_splitter = FixedTokensTextSplitter(
            chunk_size = args.chunk_size,
            chunk_overlap = args.chunk_overlap,
            separators = ["。", "？", "！", ".", "?", "!"]
        )
        text_chunks = text_splitter.get_chunks(text)
        response = ""
        for text in text_chunks:
            prompt = FINAL_MODIFY_CHUNK.format( # type: ignore
                text = text,
                tree = str(tree)
            )
            response += llm.get_response(prompt=prompt, **kwargs) if type(llm) is LLM else llm.get_response_sync(prompt=prompt, **kwargs)
    return response

async def final_modify_async(args, text:str, tree:Node, llm:AsyncLLM, **kwargs) -> str:
    if not args.enable_chunks:
        prompt = FINAL_MODIFY.format( # type: ignore
            text = text,
            tree = str(tree)
        )
        response = await llm.get_response(prompt= prompt, **kwargs)
    else:
        text_splitter = FixedTokensTextSplitter(
            chunk_size = args.chunk_size,
            chunk_overlap = args.chunk_overlap,
            separators = ["。", "？", "！", ".", "?", "!"]
        )
        text_chunks = text_splitter.get_chunks(text)
        tasks = [asyncio.ensure_future(llm.get_response(prompt=FINAL_MODIFY_CHUNK.format( # type: ignore
            text = text,
            tree = str(tree)
        ), **kwargs)) for text in text_chunks]
        responses = await asyncio.gather(*tasks)
        response = "".join(responses)
    return response

async def pipeline_async(args, text:str, overall_modification:str):
    if not args.azure:
        llm = AsyncLLM(args.chat_model, args.chat_base_url, args.chat_api_key, args.log_path)
    else:
        llm = AzureLLM(
            config_name=os.environ["CONFIG_NAME"],
            log_path=args.log_path
        )
    text_splitter = FixedTokensTextSplitter(
        chunk_size = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        separators = ["。", "？", "！", ".", "?", "!"]
    )
    llm_args = {
        "is_stream" : args.enable_stream,
        "temperature" : args.pipeline_temperature,
        "max_tokens" : args.pipeline_max_tokens,
        "log" : args.enable_log
    }
    text_chunks = text_splitter.get_chunks(text)
    entitys = get_entitys(text=overall_modification, llm=llm, **llm_args)
    entity_names = [entity['entity'] for entity in entitys]
    root = Node(text, [], None, text[:15], text[-15:], "The whole text")
    await asyncio.gather(*[get_level_nodes_async(args, text_chunk, root, llm, entity_names, 1, **llm_args) for text_chunk in text_chunks])
    tree_json = get_tree(root, args.tree_path)
    response = tree_based(overall_modification, tree_json, llm, **llm_args) if args.async_stage != 2 else await tree_based_async(overall_modification, tree_json, llm, **llm_args)
    result = final_modify(args, text, process_json_output(response), llm, **llm_args) if args.async_stage != 2 else await final_modify_async(args, text, process_json_output(response), llm, **llm_args)
    return result


def pipeline_sync(args, text:str, overall_modification:str):
    llm = LLM(args.chat_model,  args.chat_base_url, args.chat_api_key, args.log_path)
    text_splitter = FixedTokensTextSplitter(
        chunk_size = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        separators = ["。", "？", "！", ".", "?", "!"]
    )
    llm_args = {
        "is_stream" : args.enable_stream,
        "temperature" : args.pipeline_temperature,
        "max_tokens" : args.pipeline_max_tokens,
        "log" : args.enable_log
    }
    text_chunks = text_splitter.get_chunks(text)
    entitys = get_entitys(text=overall_modification, llm=llm, **llm_args)
    entity_names = [entity['entity'] for entity in entitys]
    root = Node(text, [], None, text[:15], text[-15:], "The whole text")
    for text_chunk in text_chunks:
        get_level_nodes_sync(args, text_chunk, root, llm, entity_names,1, **llm_args)
    tree_json = get_tree(root, args.tree_path)
    response = tree_based(overall_modification, tree_json, llm, **llm_args)
    result = final_modify(args, text, process_json_output(response), llm, **llm_args)
    
    return result


def main_stage_0_or_1(args):
    datas = load_datas(args.input_path)
    if args.max_samples is not None:
        datas = datas[:args.max_samples]
    final_results = []
    for data in tqdm(datas):
        result = {}
        overall_modification = data["edit_suggestion"]
        text = data["clean_text"]
        
        if args.async_stage == 1:
            pipeline = pipeline_async
        else:
            pipeline = pipeline_sync
        response, time = execute_with_retry(func = pipeline, max_retries = args.max_retries, args = args, text = text, overall_modification = overall_modification)
        result["ours_time"] = time
        if args.enable_baseline:
            if args.async_stage == 1:
                baseline = baseline_async
            else:
                baseline = baseline_sync
            baseline_response, time = execute_with_retry(func = baseline, max_retries = args.max_retries, args = args, text = text, overall_modification = overall_modification)
        else:
            baseline_response, time = None, None
        result["baseline_time"] = time
        result["ours"] = response if not isinstance(response, Errors) else None
        result["baseline"] = baseline_response if not isinstance(baseline_response, Errors) else None
        result["edit_suggestion"] = overall_modification
        result["original_text"] = text
        result["id"] = data["id"]
        result["error"] = None
        if  isinstance(response, Errors) or isinstance(baseline_response, Errors):
            result["error"] = {}
            result["error"]["pipeline_error"] = str(response) if isinstance(response, Errors) else None
            result["error"]["baseline_error"] = str(baseline_response) if isinstance(baseline_response, Errors) else None
        final_results.append(result)

    with open(args.output_path, "w") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

async def main_stage_2(args):
    datas = load_datas(args.input_path)
    if args.max_samples is not None:
        datas = datas[:args.max_samples]
    pipeline_tasks = [asyncio.ensure_future(execute_with_retry_async(func = pipeline_async, max_retries = args.max_retries, args = args, text = data["clean_text"], overall_modification = data["edit_suggestion"])) for data in datas]
    baseline_tasks = [asyncio.ensure_future(execute_with_retry_async(func = baseline_async, max_retries = args.max_retries, args = args, text = data["clean_text"], overall_modification = data["edit_suggestion"])) for data in datas] if args.enable_baseline else None
    all_tasks = pipeline_tasks + baseline_tasks if args.enable_baseline else pipeline_tasks
    print("Start running the {} tasks".format(len(all_tasks)))
    def batch_task_generator(tasks, batch_size):
        for i in range(0, len(tasks), batch_size):
            yield tasks[i:i + batch_size]
    all_results = []
    for idx, tasks in enumerate(batch_task_generator(all_tasks, args.batch_size)):
        results = await tqdm_asyncio.gather(*tasks, desc="Processing the {}/{} batch".format(idx + 1, math.ceil(len(all_tasks) / args.batch_size)))
        all_results.extend(results)
    print("Finish running the {} tasks".format(len(all_tasks)))
    pipeline_tasks_results = all_results[:len(pipeline_tasks)]
    baseline_tasks_results = all_results[len(pipeline_tasks):]
    final_results = [
        {
            "ours": pipeline_task_result if not isinstance(pipeline_task_result, Errors) else None,
            "baseline": baseline_task_result if not isinstance(baseline_task_result, Errors) else None,
            "edit_suggestion": data["edit_suggestion"],
            "original_text": data["clean_text"],
            "id": data["id"],
            "baseline_time": baseline_task_time,
            "ours_time": pipeline_task_time,
            "error": {
                "pipeline_error": str(pipeline_task_result) if isinstance(pipeline_task_result, Errors) else None,
                "baseline_error": str(baseline_task_result) if isinstance(baseline_task_result, Errors) else None
            }
        }
        for data, (pipeline_task_result, pipeline_task_time), (baseline_task_result, baseline_task_time)  in zip(datas, pipeline_tasks_results, baseline_tasks_results)
    ]
    with open(args.output_path, "w") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

def process_args(args):
    import os
    global execute_with_retry_async
    if args.work_dir is not None:
        args.output_path = os.path.join(args.work_dir, "run_output.json")
        args.log_path = os.path.join(args.work_dir, "run_log.txt")
        args.tree_path = os.path.join(args.work_dir, "run_tree.json")
    else:
        assert args.output_path is not None and args.log_path is not None and args.tree_path is not None
        args.work_dir = os.path.dirname(args.output_path)
    with open(os.path.join(args.work_dir, "run_config.json"), "w") as f:
        config = {}
        for key, value in args.__dict__.items():
            config[key] = value
        json.dump(config, f, ensure_ascii=False, indent=4)
    if args.depth_limit == 0:
        args.depth_limit = 256
    if args.max_concurrent is not None:
        assert args.async_stage == 2 and args.max_concurrent > 0 
    if args.max_rate_limit is not None:
        assert args.async_stage == 2 and args.max_rate_limit > 0
    execute_with_retry_async = concurrency_limit_with_tracking_and_rate_limit(args.max_concurrent, args.max_rate_limit)(execute_with_retry_async)
    return args


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", default=None, required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--enable_log",action="store_true")
    parser.add_argument("--chat_model", default=None)
    parser.add_argument("--chat_api_key", default=None)
    parser.add_argument("--chat_base_url", default=None)
    parser.add_argument("--enable_baseline", action="store_true")
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--language", default="en", type=str, choices=["en", "zh"])
    parser.add_argument("--max_retries", default=5, type=int)
    parser.add_argument("--tree_path", default=None, type=str)
    parser.add_argument("--work_dir", default=None, type=str)
    parser.add_argument("--depth_limit", default=2, type=int, help="The max depth of the tree")
    parser.add_argument("--chunk_size", default=4096, type=int, help="The chunk size of the text")
    parser.add_argument("--chunk_overlap", default=0, type=int, help="The chunk overlap of the text")
    parser.add_argument("--enable_chunks", action="store_true", help="Whether to enable the chunks in final modify")
    parser.add_argument("--async_stage", default=1, choices= [0, 1, 2],  type=int, help="The stage of the async pipeline 0 means all sync, 1 means the first stage is async the inner of the pipeline is async, 2 means between the pipelines is async")
    parser.add_argument("--enable_stream", action="store_true", help="Whether to enable the stream mode in the pipeline's get_level_nodes")
    parser.add_argument("--pipeline_temperature", default=0.0, type=float, help="The temperature of the pipeline")
    parser.add_argument("--pipeline_max_tokens", default=16384, type=int, help="The max tokens of the pipeline")
    parser.add_argument("--max_concurrent", default=None, type=int, help="The max concurrent of the async pipeline only for the stage 2, None means no limit")
    parser.add_argument("--max_rate_limit", default=None, type=int, help="The max rate limit of the async pipeline only for the stage 2, None means no limit, the time wait between the tasks whose unit is second")
    parser.add_argument("--batch_size", default=32, type=int, help="The batch size of the async pipeline only for the stage 2 , None means no limit")
    parser.add_argument("--azure", action="store_true", help="Whether to use the azure model")
    args = parser.parse_args()
    args = process_args(args)
    get_prompt(args.language)
    start = time.time()
    if args.async_stage == 0 or args.async_stage == 1:
        main_stage_0_or_1(args)
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_stage_2(args))
        loop.close()
    end = time.time()
    print("Total time: ", end - start)
    if args.async_stage == 2:
        print("Max concurrent_seen: ", execute_with_retry_async.tracker.max_concurrent_seen)