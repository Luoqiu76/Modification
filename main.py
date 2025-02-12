import math
from utils import *
from pipeline import Pipeline
from baseline import Baseline
from argparse import ArgumentParser
from prompt import *
from llm import LLM
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

def get_prompts(language: str)->dict:
    if language == "en":
        return {
            "extract_entities": EXTRACT_ENTITY_PROMPT_EN,
            "get_level_nodes": GET_LEVEL_NODES_PROMPT_EN,
            "find": FIND_PROMPT_EN,
            "base_tree_modification": BASE_TREE_PROMPT_EN,
            "final_modification": FINAL_MODIFY_EN,
            "baseline": BASELINE_PROMPT_EN
        }
    elif language == "zh":
        return {
            "extract_entities": EXTRACT_ENTITY_PROMPT_ZH,
            "get_level_nodes": GET_LEVEL_NODES_PROMPT_ZH,
            "find": FIND_PROMPT_ZH,
            "base_tree_modification": BASE_TREE_PROMPT_ZH,
            "final_modification": FINAL_MODIFY_ZH,
            "baseline": BASELINE_PROMPT_ZH
        }

def process_args(args):
    args.output_path = os.path.join(args.work_dir, "run_output.json")
    args.log_path = os.path.join(args.work_dir, "run_log.txt")
    config = {}
    with open(os.path.join(args.work_dir, "run_config.json"), "w", encoding="utf-8") as f:
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
    parser.add_argument("--input_path", type=str, help="Path to the input file")
    parser.add_argument("--work_dir", type=str, help="Path to the save working directory")
    parser.add_argument("--language", type=str, choices=["en", "zh"], help="Language of the input text")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to generate")
    args = parser.parse_args()
    args = process_args(args)
    datas = load_datas(args.input_path)
    datas = datas[:args.max_samples] if args.max_samples is not None else datas
    llm = LLM(args=args)
    pipeline = Pipeline(
        prompts = get_prompts(args.language),
        llm = llm
    )
    baseline = Baseline(
        llm = llm,
        prompts = get_prompts(args.language),
        llm_kwargs = get_config()['model_config']
    )
        
    if get_config()['enable_async']:
        async def main_loop():
            pipeline_tasks = [
                    asyncio.ensure_future(pipeline.forward_async(
                        text = data['clean_text'],
                        overall_modification = data['edit_suggestion']
                    )
                    )
                for data in datas
            ]
            baseline_tasks = [
                asyncio.ensure_future(
                    baseline.forward_async(
                        text = data['clean_text'],
                        overall_modification = data['edit_suggestion']
                    )
                )
                for data in datas
            ] if get_config()['enable_baseline'] else []
            tasks = pipeline_tasks + baseline_tasks
            all_tasks = pipeline_tasks + baseline_tasks if get_config()['enable_baseline'] else pipeline_tasks
            print("Start running the {} tasks".format(len(all_tasks)))
            def batch_task_generator(tasks, batch_size):
                for i in range(0, len(tasks), batch_size):
                    yield tasks[i:i + batch_size]
            all_results = []
            for idx, tasks in enumerate(batch_task_generator(all_tasks, get_config("async_config")['batch_size'])):
                results = await tqdm_asyncio.gather(*tasks, desc="Processing the {}/{} batch".format(idx + 1, math.ceil(len(all_tasks) / get_config("async_config")['batch_size'])))
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
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(final_results, f, ensure_ascii=False, indent=4, cls=Encoder)
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_loop())
    else:
        final_results = []
        for data in tqdm(datas):
            pipeline_result, pipeline_time = pipeline.forward_sync(
                text = data['clean_text'],
                overall_modification = data['edit_suggestion']
            )
            baseline_result, baseline_time = baseline.forward_sync(
                text = data['clean_text'],
                overall_modification = data['edit_suggestion']
            ) if get_config()['enable_baseline'] else None, None
            final_results.append({
                "ours": pipeline_result if not isinstance(pipeline_result, Errors) else None,
                "baseline": baseline_result if not isinstance(baseline_result, Errors) else None,
                "edit_suggestion": data["edit_suggestion"],
                "original_text": data["clean_text"],
                "id": data["id"],
                "baseline_time": baseline_time,
                "ours_time": pipeline_time,
                "error": {
                    "pipeline_error": pipeline_result if isinstance(pipeline_result, Errors) else None,
                    "baseline_error": baseline_result if isinstance(baseline_result, Errors) else None
                }
            })
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4, cls=Encoder)
            

