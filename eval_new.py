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


def get_prompt(language : str)->dict:
    import prompt
    if language == "en":
        return {
            "eval_system" : EVAL_SYSTEM_PROMPT_EN,
            "eval" : EVAL_PROMPT_EN,
            "extract_eval_result" : EXTRACT_EVAL_RESULT_PROMPT_EN
        }
    elif language == "zh":
        return {
            "eval_system" : EVAL_SYSTEM_PROMPT_ZH,
            "eval" : EVAL_PROMPT_ZH,
            "extract_eval_result" : EXTRACT_EVAL_RESULT_PROMPT_ZH
        }
        
def result_analysis(results : List[dict]):
    # judge error means the error in the judge process, model response error means the error in the model response in which the model response is None
    return_result = {
        "win": 0,
        "lose" : 0,
        "tie" : 0,
        "win_rate" : 0.0,
        "tie_rate" : 0.0,
        "win_tie_rate" : 0.0,
        "total" : 0 ,
        "judge_error" : 0,
        "model_response_error" : 0
    }
    for idx ,result in enumerate(results):
        ours = result['ours']
        baselines = result['baseline']
        result = result['result']
        # if the model response is None, then the model response error
        if ours is None or baselines is None:
            return_result["model_response_error"] += 1
            return_result["total"] += 1
            continue
        # if the result is Errors, then the judge error
        if isinstance(result, Errors):
            return_result["judge_error"] += 1
            return_result["total"] += 1
            continue

        if result["win"] == "OURS":
            return_result["win"] += 1
        elif result["win"] == "BASELINES":
            return_result["lose"] += 1
        else:
            return_result["tie"] += 1
        return_result["total"] += 1
    real_count = return_result["total"] - return_result["judge_error"] - return_result["model_response_error"] + get_config("eval_config")['epsilon'] # for epsilon smoothing
    return_result["win_rate"] = return_result["win"] / real_count
    return_result["tie_rate"] = return_result["tie"] / real_count
    return_result["win_tie_rate"] = return_result["win_rate"] + return_result["tie_rate"]
    return return_result
    


def main_sync(args):
    input_path = os.path.join(args.work_dir, "run_output.json")
    llm = LLM(
        args
    )
    final_result = {
        "baseline_pre" : [],
        "ours_pre" : []
    }
    with open(input_path, "r", encoding="utf-8") as f:
        datas = json.load(f)
    # for eval, tempature is set to 0, so the model response is deterministic
    llm_kwargs = get_config()['model_config']
    llm_kwargs['temperature'] = 0

    prompts = get_prompt(args.language)
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
                {"role" : "system", "content" : prompts["eval_system"]},  
                {"role" : "user", "content" : prompts['eval'].format(
                    original = original,
                    feedback = edit_suggestion,
                    **swap
                )}
            ]
            # firt get the model response
            response = llm.get_response_sync(
                prompt = messages,
                log_stage = "EVAL",
                **llm_kwargs
            )
            # next extract json result
            if not isinstance(response, Errors):
                messages.append({"role" : "assistant", "content" : response})
                messages.append({"role" : "user", "content" : prompts["extract_eval_result"]})
    
                result = process_json_output(
                        output = llm.get_response_sync(
                            prompt = messages,
                            log_stage = "EXTRACT_JSON",
                            **llm_kwargs
                    )
                )
            else:
                result = response
            
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
    real_count_total = baseline_pre_result["total"] + ours_pre_result["total"] - baseline_pre_result["judge_error"] - ours_pre_result["judge_error"] - baseline_pre_result["model_response_error"] - ours_pre_result["model_response_error"] + get_config("eval_config")['epsilon']
    final_result_anlysis = {
        "baseline_pre" : baseline_pre_result,
        "ours_pre" : ours_pre_result,
        "total" : {
            "win" : baseline_pre_result["win"] + ours_pre_result["win"],
            "lose" : baseline_pre_result["lose"] + ours_pre_result["lose"],
            "tie" : baseline_pre_result["tie"] + ours_pre_result["tie"],
            "win_rate" : (baseline_pre_result["win"] + ours_pre_result["win"]) / real_count_total,
            "tie_rate" : (baseline_pre_result["tie"] + ours_pre_result["tie"]) / real_count_total,
            "win_tie_rate" : (baseline_pre_result['win'] + ours_pre_result['win'] + baseline_pre_result['tie'] + ours_pre_result['tie']) / real_count_total,
        },
        "ours_time" : np.mean([data['ours_time'] for data in datas]),
        "baseline_time" : np.mean([data['baseline_time'] for data in datas]),
        "ours_baseline_time_ratio" : np.mean([data['ours_time'] for data in datas]) / np.mean([data['baseline_time'] for data in datas]),
    }
    final_result['analysis'] = final_result_anlysis
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4, cls = Encoder)

async def main_async(args):
    input_path = os.path.join(args.work_dir, "run_output.json")
    llm = LLM(
        args
    )
    # for eval, tempature is set to 0, so the model response is deterministic
    llm_kwargs = get_config()['model_config']
    llm_kwargs['temperature'] = 0

    prompts = get_prompt(args.language)

    with open(input_path, "r", encoding="utf-8") as f:
        datas = json.load(f)

    ours_pre_tasks = [asyncio.ensure_future(llm.get_response_async(
        prompt = [
            {"role" : "system", "content" : prompts['eval_system']},  
            {"role" : "user", "content" : prompts['eval'].format( 
                original = data["original_text"],
                feedback = data["edit_suggestion"],
                model1 = "OURS",
                model2 = "BASELINES",
                article1 = data["ours"],
                article2 = data["baseline"]
            )}
        ],
        log_stage = "EVAL",
        **llm_kwargs
    )) for data in datas]
    baseline_pre_tasks = [asyncio.ensure_future(llm.get_response_async(
        prompt = [
            {"role" : "system", "content" : prompts['eval_system']},
            {"role" : "user", "content" : prompts['eval'].format(
                original = data["original_text"],
                feedback = data["edit_suggestion"],
                model1 = "BASELINES",
                model2 = "OURS",
                article1 = data["baseline"],
                article2 = data["ours"]
            )}
        ],
        log_stage = "EVAL",
        **llm_kwargs
    )) for data in datas]
    def batch_task_generator(tasks, batch_size):
        for i in range(0, len(tasks), batch_size):
            yield tasks[i:i + batch_size]
    all_tasks = ours_pre_tasks + baseline_pre_tasks
    all_results = []
    for idx, tasks in enumerate(batch_task_generator(all_tasks, get_config("async_config")['batch_size'])):
        results = await tqdm_asyncio.gather(*tasks, desc="Processing the {}/{} batch in evaluating models".format(idx + 1, math.ceil(len(all_tasks) / get_config("async_config")['batch_size'])))
        all_results.extend(results)
    # all_results = [result for (result, _) in all_results]
    ours_pre_results = all_results[:len(ours_pre_tasks)]
    baseline_pre_results = all_results[len(ours_pre_tasks):]
    async def extract_json_result(pre_messages : List, response : Union[str, Errors], data: dict) -> Union[dict, Errors]:
        # judge error
        if isinstance(response, Errors):
            return response
        messages = pre_messages + [
            {"role" : "assistant", "content" : response},
            {"role" : "user", "content" : prompts['extract_eval_result']}
        ]
        try:
            return process_json_output(
                output = await llm.get_response_async(
                    prompt = messages,
                    log_stage = "EXTRACT_JSON",
                    **llm_kwargs
                )
            )
        except Exception as e:
            return Errors(
                [
                    (e, repr(traceback.format_exc()))
                ]
            )
        
    # extract_ours_pre_results_tasks = [asyncio.ensure_future(llm.get_response_async(
    #     prompt = [
    #         {"role" : "system", "content" : prompts['eval_system']},
    #         {"role" : "user", "content" : prompts['eval'].format(
    #             original = data["original_text"],
    #             feedback = data["edit_suggestion"],
    #             model1 = "OURS",
    #             model2 = "BASELINES",
    #             article1 = data["ours"],
    #             article2 = data["baseline"]
    #         )},
    #         {"role" : "assistant", "content" : str(result)}, # if the first response isinstance(Errors) then the result is str, can be processed in anlysis function
    #         {"role" : "user", "content" : prompts['extract_eval_result']}
    #     ],
    #     log_stage = "EXTRACT_JSON",
    #     **llm_kwargs
    # )) for data, result in zip(datas, ours_pre_results)]
    extract_ours_pre_results_tasks = [asyncio.ensure_future(extract_json_result(
        pre_messages = [
            {"role" : "system", "content" : prompts['eval_system']},
            {"role" : "user", "content" : prompts['eval'].format(
                original = data["original_text"],
                feedback = data["edit_suggestion"],
                model1 = "OURS",
                model2 = "BASELINES",
                article1 = data["ours"],
                article2 = data["baseline"]
            )}
        ],
        response = result,
        data = data
    )) for data, result in zip(datas, ours_pre_results)]
    # extract_baseline_pre_results_tasks = [asyncio.ensure_future(llm.get_response_async(
    #     prompt = [
    #         {"role" : "system", "content" : prompts['eval_system']},
    #         {"role" : "user", "content" : prompts['eval'].format(
    #             original = data["original_text"],
    #             feedback = data["edit_suggestion"],
    #             model1 = "BASELINES",
    #             model2 = "OURS",
    #             article1 = data["baseline"],
    #             article2 = data["ours"]
    #         )},
    #         {"role" : "assistant", "content" : str(result)}, # if the first response isinstance(Errors) then the result is str, can be processed in anlysis function
    #         {"role" : "user", "content" : prompts['extract_eval_result']}
    #     ],
    #     log_stage = "EXTRACT_JSON",
    #     **llm_kwargs
    # )) for data, result in zip(datas, baseline_pre_results)]
    extract_baseline_pre_results_tasks = [asyncio.ensure_future(extract_json_result(
        pre_messages = [
            {"role" : "system", "content" : prompts['eval_system']},
            {"role" : "user", "content" : prompts['eval'].format(
                original = data["original_text"],
                feedback = data["edit_suggestion"],
                model1 = "BASELINES",
                model2 = "OURS",
                article1 = data["baseline"],
                article2 = data["ours"]
            )}
        ],
        response = result,
        data = data
    )) for data, result in zip(datas, baseline_pre_results)]
    all_tasks = extract_ours_pre_results_tasks + extract_baseline_pre_results_tasks
    all_results = []
    for idx, tasks in enumerate(batch_task_generator(all_tasks, get_config("async_config")['batch_size'])):
        results = await tqdm_asyncio.gather(*tasks, desc="Processing the {}/{} batch in extracting result".format(idx + 1, math.ceil(len(all_tasks) / get_config("async_config")['batch_size'])))
        all_results.extend(results)
    # all_results = [result for (result, _) in all_results]
    ours_pre_results_final = all_results[:len(extract_ours_pre_results_tasks)]
    baseline_pre_results_final = all_results[len(extract_ours_pre_results_tasks):]
    # [0] means only the result not the time is reserved
    
    # ours_pre_results_final = [process_json_output(output = result) for result in ours_pre_results_final]
    # baseline_pre_results_final = [process_json_output(output = result) for result in baseline_pre_results_final]
    # tmp_ours_pre_results_final = []
    # tmp_baseline_pre_results_final = []
    # for result in ours_pre_results_final:
    #     try:
    #         tmp_ours_pre_results_final.append(process_json_output(output = result))
    #     except Exception as e:
    #         tmp_ours_pre_results_final.append(Errors(
    #             [
    #                 (e, repr(traceback.format_exc()))
    #             ]
    #         ))
    # for result in baseline_pre_results_final:
    #     try:
    #         tmp_baseline_pre_results_final.append(process_json_output(output = result))
    #     except Exception as e:
    #         tmp_baseline_pre_results_final.append(Errors(
    #             [
    #                 (e, repr(traceback.format_exc()))
    #             ]
    #         ))
    # ours_pre_results_final = tmp_ours_pre_results_final
    # baseline_pre_results_final = tmp_baseline_pre_results_final
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
            "result" : baseline_pre_result_final,
        })
        final_result['ours_pre'].append({
            "original" : data["original_text"],
            "edit_suggestion" : data["edit_suggestion"],
            "ours" : data["ours"],
            "baseline" : data["baseline"],
            "response" : ours_pre_result,
            "result" : ours_pre_result_final,
        })
    baseline_pre_result = result_analysis(final_result["baseline_pre"])
    ours_pre_result = result_analysis(final_result["ours_pre"])
    real_count_total = baseline_pre_result["total"] + ours_pre_result["total"] - baseline_pre_result["judge_error"] - ours_pre_result["judge_error"] - baseline_pre_result["model_response_error"] - ours_pre_result["model_response_error"] + get_config("eval_config")['epsilon']
    final_result_anlysis = {
        "baseline_pre" : baseline_pre_result,
        "ours_pre" : ours_pre_result,
        "total" : {
            "win" : baseline_pre_result["win"] + ours_pre_result["win"],
            "lose" : baseline_pre_result["lose"] + ours_pre_result["lose"],
            "tie" : baseline_pre_result["tie"] + ours_pre_result["tie"],
            "win_rate" : (baseline_pre_result["win"] + ours_pre_result["win"]) / (real_count_total + get_config("eval_config")['epsilon']),
            "tie_rate" : (baseline_pre_result["tie"] + ours_pre_result["tie"]) / (real_count_total + get_config("eval_config")['epsilon']),
            "win_tie_rate" : (baseline_pre_result["win"] + ours_pre_result["win"] + baseline_pre_result["tie"] + ours_pre_result["tie"]) / (real_count_total + get_config("eval_config")['epsilon']),
        },
        "ours_time" : np.mean([data['ours_time'] for data in datas] if all([data['ours_time'] is not None for data in datas]) else 0),
        "baseline_time" : np.mean([data['baseline_time'] for data in datas] if all([data['baseline_time'] is not None for data in datas]) else 0),
        "ours_baseline_time_ratio" : np.mean([data['ours_time'] for data in datas]) / np.mean([data['baseline_time'] for data in datas]) if all([data['baseline_time'] is not None and  data['ours_time'] is not None for data in datas]) else 0,
    }
    final_result['analysis'] = final_result_anlysis
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, cls = Encoder, ensure_ascii=False, indent=4)




def process_args(args):
    # process work_dir and add output_path, log_path,
    args.output_path = os.path.join(args.work_dir, "eval_output.json")
    args.log_path = os.path.join(args.work_dir, "eval_log.txt")
    config = {}
    with open(os.path.join(args.work_dir, "eval_config.json"), "w", encoding='utf-8') as f:
        for key, value in vars(args).items():
            config[key] = value
        disk_config = get_config()
        for key, value in disk_config.items():
            if key not in config:
                config[key] = value
        config['api_key'] = "NOT PRINTED"
        json.dump(config, f, ensure_ascii=False, indent=4)
    return args






if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--language", default="en", type=str, choices=["en", "zh"])
    parser.add_argument("--work_dir", default=None, type=str, required=True)

    args = parser.parse_args()
    args = process_args(args)
    if get_config()['enable_async']:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_async(args))
    else:
        main_sync(args)




