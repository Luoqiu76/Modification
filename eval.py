import math
import os
from llm import *
import json
from typing import List, Tuple,TypedDict
from argparse import ArgumentParser
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from prompt import *
import numpy as np
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
        
def result_analysis(results : List[dict]):
    # judge error means the error in the judge process, model response error means the error in the model response in which the model response is None
    return_result = {
        "win": 0,
        "lose" : 0,
        "tie" : 0,
        "win_rate" : 0.0,
        "tie_rate" : 0.0,
        "win_tie_rate" : 0.0,
        "errors" : [],
        "total" : 0 ,
        "judge_error" : 0,
        "model_response_error" : 0
    }
    for idx ,result in enumerate(results):
        ours = result['ours']
        baselines = result['baseline']
        result = result['result']
        if ours is None or baselines is None:
            return_result["model_response_error"] += 1
            return_result["total"] += 1
            continue

        try:
            if result["win"] == "OURS":
                return_result["win"] += 1
            elif result["win"] == "BASELINES":
                return_result["lose"] += 1
            else:
                return_result["tie"] += 1
            return_result["total"] += 1
        except Exception as e:
            return_result['errors'].append({
                "error_result" : result,
                "error_idx" : idx,
                "error_message" : str(e)
            })
            return_result['judge_error'] += 1

    return_result["win_rate"] = return_result["win"] / (return_result["total"] - return_result["judge_error"] - return_result["model_response_error"] + args.epsilon)
    return_result["tie_rate"] = return_result["tie"] / (return_result["total"] - return_result["judge_error"] - return_result["model_response_error"] + args.epsilon)
    return_result["win_tie_rate"] = return_result["win_rate"] + return_result["tie_rate"]
    return return_result
    


def main_sync(args):
    input_path = os.path.join(args.work_dir, "run_output.json")
    llm_args = {
        "max_tokens" : 1024,
        "is_stream" : args.enable_stream,
        "temperature" : 0,
        "log" : False
    }
    llm = LLM(
        args.chat_model,
        args.chat_base_url,
        args.chat_api_key,
        log_path=args.log_path,
    )
    final_result = {
        "baseline_pre" : [],
        "ours_pre" : []
    }
    with open(input_path, "r", encoding="utf-8") as f:
        datas = json.load(f)

    for data in tqdm(datas):
        original = data["original_text"]
        edit_suggestion = data["edit_suggestion"]
        result_ours = data["ours"]
        result_baselines = data["baseline"]
        swap_list = [
            {
                "model1" : "BASELINES",
                "model2" : "OURS",
                "article1" : result_baselines,
                "article2" : result_ours
            },
            {
                "model1" : "OURS",
                "model2" : "BASELINES",
                "article1" : result_ours,
                "article2" : result_baselines
            }
        ]
        for swap in swap_list:
            messages = [
                {"role" : "system", "content" : EVAL_SYSTEM_PROMPT},  # type: ignore
                {"role" : "user", "content" : EVAL_PROMPT.format( # type: ignore
                    original = original,
                    feedback = edit_suggestion,
                    **swap
                )}
            ]
            # firt get the model response
            response = execute_with_retry(
                llm.get_response,
                args.max_retries,
                messages,
                **llm_args
            )
            # next extract json result
            if not isinstance(response, Errors):
                messages.append({"role" : "assistant", "content" : response})
                messages.append({"role" : "user", "content" : EXTRACT_EVAL_RESULT_PROMPT}) # type: ignore
                result = execute_with_retry(
                    process_json_output,
                    args.max_retries,
                    enable_wait=False,
                    output = llm.get_response(
                        messages,
                        max_tokens=1024,
                        is_stream=True,
                        temperature=0,
                        log=False
                    )
                )
            else:
                result = response
            if isinstance(result, Errors):
                result = str(result)
            
            
            final_result['baseline_pre'].append({
                "original" : original,
                "edit_suggestion" : edit_suggestion,
                "ours" : result_ours,
                "baseline" : result_baselines,
                "response" : response,
                "result" : result
            }) if swap["model1"] == "BASELINES" else final_result['ours_pre'].append({
                "original" : original,
                "edit_suggestion" : edit_suggestion,
                "ours" : result_ours,
                "baseline" : result_baselines,
                "response" : response,
                "result" : result
            })
    baseline_pre_result = result_analysis(final_result["baseline_pre"])
    ours_pre_result = result_analysis(final_result["ours_pre"])

    final_result_anlysis = {
        "baseline_pre" : baseline_pre_result,
        "ours_pre" : ours_pre_result,
        "total" : {
            "win" : baseline_pre_result["win"] + ours_pre_result["win"],
            "lose" : baseline_pre_result["lose"] + ours_pre_result["lose"],
            "tie" : baseline_pre_result["tie"] + ours_pre_result["tie"],
            "win_rate" : (baseline_pre_result["win"] + ours_pre_result["win"]) / (baseline_pre_result["total"] + ours_pre_result["total"] - baseline_pre_result["judge_error"] - ours_pre_result["judge_error"] - baseline_pre_result["model_response_error"] - ours_pre_result["model_response_error"] + args.epsilon),
            "tie_rate" : (baseline_pre_result["tie"] + ours_pre_result["tie"]) / (baseline_pre_result["total"] + ours_pre_result["total"] - baseline_pre_result["judge_error"] - ours_pre_result["judge_error"] - baseline_pre_result["model_response_error"] - ours_pre_result["model_response_error"] + args.epsilon),
            "win_tie_rate" : (baseline_pre_result["win"] + ours_pre_result["win"] + baseline_pre_result["tie"] + ours_pre_result["tie"]) / (baseline_pre_result["total"] + ours_pre_result["total"] - baseline_pre_result["judge_error"] - ours_pre_result["judge_error"] - baseline_pre_result["model_response_error"] - ours_pre_result["model_response_error"] + args.epsilon),
        },
        "ours_time" : np.mean([data['ours_time'] for data in datas]),
        "baseline_time" : np.mean([data['baseline_time'] for data in datas]),
        "ours_baseline_time_ratio" : np.mean([data['ours_time'] for data in datas]) / np.mean([data['baseline_time'] for data in datas]),
    }
    final_result['analysis'] = final_result_anlysis
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)

async def main_async(args):
    input_path = os.path.join(args.work_dir, "run_output.json")
    llm_args = {
        "max_tokens" : 1024,
        "is_stream" : args.enable_stream,
        "temperature" : 0,
        "log" : False
    }
    if args.enable_azure:
        llm = AzureLLM(
            config_name=os.environ["JUDGE_CONFIG_NAME"],
            log_path=args.log_path,
        )
    else:
        llm = AsyncLLM(
            args.chat_model,
            args.chat_base_url,
            args.chat_api_key,
            log_path=args.log_path,
        )
    with open(input_path, "r", encoding="utf-8") as f:
        datas = json.load(f)
    # ours_pre_tasks = [asyncio.ensure_future(llm.get_response(prompt = [
    #     {"role" : "system", "content" : EVAL_SYSTEM_PROMPT},  # type: ignore
    #     {"role" : "user", "content" : EVAL_PROMPT.format( # type: ignore
    #         original = data["original_text"],
    #         feedback = data["edit_suggestion"],
    #         model1 = "OURS",
    #         model2 = "BASELINES",
    #         article1 = data["ours"],
    #         article2 = data["baseline"]
    #     )}
    # ], max_tokens=1024, is_stream=True, temperature=0, log=False)) for data in datas]
    ours_pre_tasks = [asyncio.ensure_future(execute_with_retry_async(
        func = llm.get_response,
        max_retries = args.max_retries,
        prompt = [
            {"role" : "system", "content" : EVAL_SYSTEM_PROMPT},  # type: ignore
            {"role" : "user", "content" : EVAL_PROMPT.format( # type: ignore
                original = data["original_text"],
                feedback = data["edit_suggestion"],
                model1 = "OURS",
                model2 = "BASELINES",
                article1 = data["ours"],
                article2 = data["baseline"]
            )}
        ],
        **llm_args
    )) for data in datas]
    # baseline_pre_tasks = [asyncio.ensure_future(llm.get_response(prompt = [
    #     {"role" : "system", "content" : EVAL_SYSTEM_PROMPT},  # type: ignore
    #     {"role" : "user", "content" : EVAL_PROMPT.format( # type: ignore
    #         original = data["original_text"],
    #         feedback = data["edit_suggestion"],
    #         model1 = "BASELINES",
    #         model2 = "OURS",
    #         article1 = data["baseline"],
    #         article2 = data["ours"]
    #     )}
    # ], max_tokens=1024, is_stream=True, temperature=0, log=False)) for data in datas]
    baseline_pre_tasks = [asyncio.ensure_future(execute_with_retry_async(
        func = llm.get_response,
        max_retries = args.max_retries,
        prompt = [
            {"role" : "system", "content" : EVAL_SYSTEM_PROMPT},  # type: ignore
            {"role" : "user", "content" : EVAL_PROMPT.format( # type: ignore
                original = data["original_text"],
                feedback = data["edit_suggestion"],
                model1 = "BASELINES",
                model2 = "OURS",
                article1 = data["baseline"],
                article2 = data["ours"]
            )}
        ],
        **llm_args
    )) for data in datas]
    def batch_task_generator(tasks, batch_size):
        for i in range(0, len(tasks), batch_size):
            yield tasks[i:i + batch_size]
    all_tasks = ours_pre_tasks + baseline_pre_tasks
    all_results = []
    for idx, tasks in enumerate(batch_task_generator(all_tasks, args.batch_size)):
        results = await tqdm_asyncio.gather(*tasks, desc="Processing the {}/{} batch in evaluating models".format(idx + 1, math.ceil(len(all_tasks) / args.batch_size)))
        all_results.extend(results)
    all_results = [result for (result, _) in all_results]

    ours_pre_results = all_results[:len(ours_pre_tasks)]
    baseline_pre_results = all_results[len(ours_pre_tasks):]
    # extract_ours_pre_results_tasks = [asyncio.ensure_future(llm.get_response(prompt = [
    #     {"role" : "system", "content" : EVAL_SYSTEM_PROMPT},  # type: ignore
    #     {"role" : "user", "content" : EVAL_PROMPT.format(  # type: ignore
    #         original = data["original_text"],
    #         feedback = data["edit_suggestion"],
    #         model1 = "OURS",
    #         model2 = "BASELINES",
    #         article1 = data["ours"],
    #         article2 = data["baseline"]
    #     )},          
    #     {"role" : "assistant", "content" : result},
    #     {"role" : "user", "content" : EXTRACT_EVAL_RESULT_PROMPT} # type: ignore
    # ], max_tokens=1024, is_stream=True, temperature=0, log=False)) for data, result in zip(datas, ours_pre_results)]
    extract_ours_pre_results_tasks = [asyncio.ensure_future(execute_with_retry_async(
        func = llm.get_response,
        max_retries = args.max_retries,
        prompt = [
            {"role" : "system", "content" : EVAL_SYSTEM_PROMPT},  # type: ignore
            {"role" : "user", "content" : EVAL_PROMPT.format(  # type: ignore
                original = data["original_text"],
                feedback = data["edit_suggestion"],
                model1 = "OURS",
                model2 = "BASELINES",
                article1 = data["ours"],
                article2 = data["baseline"]
            )},
            {"role" : "assistant", "content" : str(result)}, # if the first response isinstance(Errors) then the result is str, can be processed in anlysis function
            {"role" : "user", "content" : EXTRACT_EVAL_RESULT_PROMPT} # type: ignore
        ],
        **llm_args
    )) for data, result in zip(datas, ours_pre_results)]
    # extract_baseline_pre_results_tasks = [asyncio.ensure_future(llm.get_response(prompt = [
    #     {"role" : "system", "content" : EVAL_SYSTEM_PROMPT},  # type: ignore
    #     {"role" : "user", "content" : EVAL_PROMPT.format( # type: ignore
    #         original = data["original_text"],
    #         feedback = data["edit_suggestion"],
    #         model1 = "BASELINES",
    #         model2 = "OURS",
    #         article1 = data["baseline"],
    #         article2 = data["ours"]
    #     )},           
    #     {"role" : "assistant", "content" : result},
    #     {"role" : "user", "content" : EXTRACT_EVAL_RESULT_PROMPT} # type: ignore
    # ], max_tokens=1024, is_stream=True, temperature=0, log=False)) for data, result in zip(datas, baseline_pre_results)]
    extract_baseline_pre_results_tasks = [asyncio.ensure_future(execute_with_retry_async(
        func = llm.get_response,
        max_retries = args.max_retries,
        prompt = [
            {"role" : "system", "content" : EVAL_SYSTEM_PROMPT},  # type: ignore
            {"role" : "user", "content" : EVAL_PROMPT.format( # type: ignore
                original = data["original_text"],
                feedback = data["edit_suggestion"],
                model1 = "BASELINES",
                model2 = "OURS",
                article1 = data["baseline"],
                article2 = data["ours"]
            )},
            {"role" : "assistant", "content" : str(result)}, # if the first response isinstance(Errors) then the result is str, can be processed in anlysis function
            {"role" : "user", "content" : EXTRACT_EVAL_RESULT_PROMPT} # type: ignore
        ],
        **llm_args
    )) for data, result in zip(datas, baseline_pre_results)]
    all_tasks = extract_ours_pre_results_tasks + extract_baseline_pre_results_tasks

    all_results = []
    for idx, tasks in enumerate(batch_task_generator(all_tasks, args.batch_size)):
        results = await tqdm_asyncio.gather(*tasks, desc="Processing the {}/{} batch in extracting result".format(idx + 1, math.ceil(len(all_tasks) / args.batch_size)))
        all_results.extend(results)
    all_results = [result for (result, _) in all_results]
    ours_pre_results_final = all_results[:len(extract_ours_pre_results_tasks)]
    baseline_pre_results_final = all_results[len(extract_ours_pre_results_tasks):]
    # [0] means only the result not the time is reserved
    ours_pre_results_final = [execute_with_retry(process_json_output, args.max_retries, enable_wait = False, output = result)[0] for result in ours_pre_results_final]
    baseline_pre_results_final = [execute_with_retry(process_json_output, args.max_retries, enable_wait = False, output = result)[0] for result in baseline_pre_results_final]
    final_result = {
        "baseline_pre" : [],
        "ours_pre" : []
    }
    # str the result, if the result is Errors, then the result is str for json serializable
    for data, ours_pre_result, baseline_pre_result, ours_pre_result_final, baseline_pre_result_final in zip(datas, ours_pre_results, baseline_pre_results, ours_pre_results_final, baseline_pre_results_final):
        final_result['baseline_pre'].append({
            "original" : data["original_text"],
            "edit_suggestion" : data["edit_suggestion"],
            "ours" : data["ours"],
            "baseline" : data["baseline"],
            "response" : baseline_pre_result,
            "result" : baseline_pre_result_final if not isinstance(baseline_pre_result_final, Errors) else str(baseline_pre_result_final),
        })
        final_result['ours_pre'].append({
            "original" : data["original_text"],
            "edit_suggestion" : data["edit_suggestion"],
            "ours" : data["ours"],
            "baseline" : data["baseline"],
            "response" : ours_pre_result,
            "result" : ours_pre_result_final if not isinstance(ours_pre_result_final, Errors) else str(ours_pre_result_final),
        })
    baseline_pre_result = result_analysis(final_result["baseline_pre"])
    ours_pre_result = result_analysis(final_result["ours_pre"])
    final_result_anlysis = {
        "baseline_pre" : baseline_pre_result,
        "ours_pre" : ours_pre_result,
        "total" : {
            "win" : baseline_pre_result["win"] + ours_pre_result["win"],
            "lose" : baseline_pre_result["lose"] + ours_pre_result["lose"],
            "tie" : baseline_pre_result["tie"] + ours_pre_result["tie"],
            "win_rate" : (baseline_pre_result["win"] + ours_pre_result["win"]) / (baseline_pre_result["total"] + ours_pre_result["total"] - baseline_pre_result["judge_error"] - ours_pre_result["judge_error"] - baseline_pre_result["model_response_error"] - ours_pre_result["model_response_error"] + args.epsilon),
            "tie_rate" : (baseline_pre_result["tie"] + ours_pre_result["tie"]) / (baseline_pre_result["total"] + ours_pre_result["total"] - baseline_pre_result["judge_error"] - ours_pre_result["judge_error"] - baseline_pre_result["model_response_error"] - ours_pre_result["model_response_error"] + args.epsilon),
            "win_tie_rate" : (baseline_pre_result["win"] + ours_pre_result["win"] + baseline_pre_result["tie"] + ours_pre_result["tie"]) / (baseline_pre_result["total"] + ours_pre_result["total"] - baseline_pre_result["judge_error"] - ours_pre_result["judge_error"] - baseline_pre_result["model_response_error"] - ours_pre_result["model_response_error"] + args.epsilon),
        },
        "ours_time" : np.mean([data['ours_time'] for data in datas] if all([data['ours_time'] is not None for data in datas]) else 0),
        "baseline_time" : np.mean([data['baseline_time'] for data in datas] if all([data['baseline_time'] is not None for data in datas]) else 0),
        "ours_baseline_time_ratio" : np.mean([data['ours_time'] for data in datas]) / np.mean([data['baseline_time'] for data in datas]) if all([data['baseline_time'] is not None and  data['ours_time'] is not None for data in datas]) else 0,
    }
    final_result['analysis'] = final_result_anlysis
    class ErrorsEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Errors):
                return obj.__str__()
            return json.JSONEncoder.default(self, obj)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, cls = ErrorsEncoder, ensure_ascii=False, indent=4)




def process_args(args):
    import os
    if args.work_dir is not None:
        args.output_path = os.path.join(args.work_dir, "eval_result.json")
        args.log_path = os.path.join(args.work_dir, "eval_log.txt")
    else:
        assert args.output_path is not None and args.log_path is not None
        args.work_dir = os.path.dirname(args.output_path)
    
    with open(os.path.join(args.work_dir, "eval_config.json"), "w") as f:
        config = {}
        for key, value in args.__dict__.items():
            config[key] = value
        json.dump(config, f, ensure_ascii=False, indent=4)
    if args.max_concurrent is not None:
        assert args.enable_async is True
    if args.max_rate_limit is not None:
        assert args.enable_async is True
    global execute_with_retry_async
    execute_with_retry_async = concurrency_limit_with_tracking_and_rate_limit(args.max_concurrent, args.max_rate_limit)(execute_with_retry_async)
    return args






if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--chat_model", default=None)
    parser.add_argument("--chat_api_key", default=None)
    parser.add_argument("--chat_base_url", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--language", default="en", type=str, choices=["en", "zh"])
    parser.add_argument("--max_retries", default=5, type=int)
    parser.add_argument("--epsilon", default=1e-5, type=float)
    parser.add_argument("--work_dir", default=None)
    parser.add_argument("--enable_async", action="store_true")
    parser.add_argument("--enable_stream", action="store_true")
    parser.add_argument("--max_concurrent", default=None, type=int, help="The max concurrent of the async , None means no limit")
    parser.add_argument("--max_rate_limit", default=None, type=int, help="The max rate limit of the async , None means no limit, the time wait between the tasks whose unit is second")
    parser.add_argument("--batch_size", default=32, type=int, help="The batch size of the async pipeline only for the stage 2 , None means no limit")
    parser.add_argument("--enable_azure", action="store_true")
    args = parser.parse_args()
    args = process_args(args)
    get_prompt(args.language)
    if args.enable_async:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_async(args))
        loop.close()
        print("Max concurrent seen: ", execute_with_retry_async.tracker.max_concurrent_seen)
    else:
        main_sync(args)




