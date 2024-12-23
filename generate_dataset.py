from argparse import ArgumentParser
from contextlib import contextmanager
import random
import re
import time
from typing import Tuple
from llm import *
import pyarrow.parquet as pq
from pandas import DataFrame
from prompt import *
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import json
from bs4 import BeautifulSoup
import pandas as pd
from bs4 import UnicodeDammit
from errors import Errors
import os
from utils import *
import math
def trunk_string(args, string, max_length):
    language = args.language
    if max_length is None:
        return string
    if language == "en":
        word_list = string.split(" ")[:max_length]
        return " ".join(word_list)
    else:
        return string[:max_length]


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
    globals()['language'] = language.lower()





async def get_quality_datas(args, llm, data_paths: List[str], output_path, max_length, **llm_args):
    language = args.language
    remain_samples = args.max_samples
    def duplicate_datas(datas):
        article_ids = [data["article_id"] for data in datas] # 有重复的，一篇文章有两个作者分别提问题，总长度为230则有115个article_id
        article_ids = list(set(article_ids))
        article_ids_map = {article_id : False for article_id in article_ids}
        duplicate_datas = []
        for data in datas:
            article_id = data["article_id"]
            if article_ids_map[article_id] is False:
                article_ids_map[article_id] = True
                duplicate_datas.append(data)
            else:
                continue
        datas = duplicate_datas
        return datas
    

    def truncate_datas(datas, remain_samples):
        datas_len = min(len(datas), remain_samples) if remain_samples is not None else len(datas)
        if datas_len <= 0:
            return [], 0
        datas = datas[:datas_len]
        remain_samples = remain_samples - datas_len   
        return datas, remain_samples
    

    async def get_quality_data(data, data_path):
        article_id = data["article_id"]
        article = data["article"]
        metadata = [
            {
                "source" : data["source"],
                "explaination" : "The source of the article" if language == "en" else "文章来源"
            },
            {
                "author" : data["author"],
                "explaination" : "The author of the article" if language == "en" else "文章作者"
            },
            {
                "topic" : data["topic"],
                "explaination" : "The topic of the article" if language == "en" else "文章主题"
            },
            [
                {
                    "question" : question['question'],
                    "options" : question['options'],
                }
                for question in data["questions"]
            ] + [
                {
                    "explaination" : "The questions and options of the article. This data is from a qa dataset. The questions and options can help you to understand the article better and generate a better edit suggestion." if language == "en" else "文章的问题和选项。这些数据来自一个qa数据集。问题和选项可以帮助您更好地理解文章并生成更好的编辑建议。"
                }
            ]
        ]
        article = trunk_string(args, article, max_length)

        generate_quality_prompt = GENERATE_EDIT_PROMPT.format( # type: ignore
            text = article,
            metadata = str(metadata)
        )
        response = await llm.get_response(generate_quality_prompt, **llm_args)
        result = {
            "clean_text" : article,
            "edit_suggestion" : response,
            "id" : data_path.split("/")[-1] + "_" + str(article_id),
        }
        with open(output_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    processed_datas = []
    for data_path in data_paths:
        with open(data_path, "r", encoding='utf-8') as f:
            datas = f.readlines()
            datas = [json.loads(data) for data in datas]
        datas = duplicate_datas(datas)
        if remain_samples is not None:
            if remain_samples <= 0:
                break
            else:
                datas, remain_samples = truncate_datas(datas, remain_samples)
                processed_datas.append({
                    "datas" : datas,
                    "datas_path" : data_path
                })
        else:
            processed_datas.append({
                "datas" : datas,
                "datas_path" : data_path
            })
    tasks = []
    for processed_data in processed_datas:
        datas = processed_data["datas"]
        data_path = processed_data["datas_path"]
        tasks.extend([get_quality_data(data, data_path) for data in datas])
    await tqdm_asyncio.gather(*tasks, task = "Process quality datas")
    args.max_samples = remain_samples
    return


async def get_narrative_qa_datas(args, llm, data_paths: List[str],output_path, max_length, **llm_args):
    assert len(data_paths) == 1
    data_dir = data_paths[0]
    tmp_dir = os.path.join(data_dir, "tmp")
    summary_path = os.path.join(data_dir, "third_party/wikipedia/summaries.csv")
    summary_df = pd.read_csv(summary_path)
    summary_df.set_index("document_id", inplace=True)
    qaps_df = pd.read_csv(os.path.join(data_dir, "qaps.csv"))
    documents_df = pd.read_csv(os.path.join(data_dir, "documents.csv"))
    documents_df.set_index("document_id", inplace=True)
    language = args.language
    remain_samples = args.max_samples
    tasks = []
    async def get_narrative_qa_data(document_id, article, metadata):
        generate_naive_qa_prompt = GENERATE_EDIT_PROMPT.format( # type: ignore
                text = article,
                metadata = str(metadata)
            )
        response = await llm.get_response(generate_naive_qa_prompt, **llm_args)
        set_name = documents_df.loc[document_id]["set"]
        result = {
            "clean_text" : article,
            "edit_suggestion" : response,
            "id" : document_id + "_" + set_name,
        }
        with open(output_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    file_text_dict = {
        file : ""
        for file in os.listdir(tmp_dir) if file.endswith(".content")
    }
    async def read_file(file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            text = f.read()
            file_text_dict[file_path.split("/")[-1]] = text
    await tqdm_asyncio.gather(*[read_file(os.path.join(tmp_dir, file)) for file in os.listdir(tmp_dir) if file.endswith(".content")], desc="Read files")
    for file in os.listdir(tmp_dir):
        if file.endswith(".content"):
            text = file_text_dict[file]
            if "<pre>" in text:
                start = text.find("<pre>")
                end = text.find("</pre>")
                spilt_text = text[start + len("<pre>"):end]
                soup = BeautifulSoup(spilt_text, "html.parser")
                article = soup.get_text()
                if len(spilt_text) / len(text) < 0.9:
                    continue
            else:
                assert "<html>" not in text
                article = text
            article = trunk_string(args, article, max_length)
            if remain_samples is not None:
                if remain_samples <= 0:
                    break
                else:
                    remain_samples -= 1
            document_id = file.split(".")[0]
            summary = summary_df.loc[document_id]["summary"]
            qaps = qaps_df.query(f"document_id == '{document_id}'")
            qaps = [{
                "question" : row["question"],
                "answer1" : row["answer1"],
                "answer2" : row["answer2"]
            }
            for _, row in qaps.iterrows()] + [
                {
                    "explaination" : "The questions and answers of the article. This data is from a narrative qa dataset. Both answer1 and answer2 are correct answers to the question, but they are phrased differently. The questions and answers can help you to understand the article better and generate a better edit suggestion." if language == "en" else "文章的问题和答案。这些数据来自一个narrative qa数据集。answer1和answer2都是问题的正确答案，但表述不同。问题和答案可以帮助您更好地理解文章并生成更好的编辑建议。"
                }
            ]
            metadata = [
            {
                "summary" : summary,
                "explaination" : "The summary of the article" if language == "en" else "文章的摘要"
            },
            qaps
            ]
            tasks.append(execute_with_retry_async(get_narrative_qa_data, document_id=document_id, article=article, metadata=metadata))
            
    await tqdm_asyncio.gather(*tasks, desc="Process narrative qa datas")
    args.max_samples = remain_samples
    return
    

async def get_lv_eval_datas(args, llm, data_paths: List[str], output_path, max_length, **llm_args):
    assert len(data_paths) == 1
    data_path = data_paths[0]
    dataset_name_list = [
        # "cmrc_mixup",
        # "dureader_mixup",
        # "factrecall_en",
        # "factrecall_zh",
        # "hotpotwikiqa_mixup",
        # "lic_mixup",
        "loogle_CR_mixup",
        "loogle_MIR_mixup",
        "loogle_SD_mixup",
        "multifieldqa_en_mixup",
        "multifieldqa_zh_mixup"
    ]
    dataset_dir_list = [
        os.path.join(data_path, dataset_name)
        for dataset_name in dataset_name_list
    ]
    dataset_path_list = [
        os.path.join(dataset_dir, dataset_name + "_256k.jsonl")
        for dataset_dir, dataset_name in zip(dataset_dir_list, dataset_name_list)
    ]
    output_path_list = [
        os.path.join(args.output_path, dataset_name + "_256k.jsonl")
        for dataset_name in dataset_name_list
    ]
    # clear the file
    for output_path in output_path_list:
        if os.path.exists(output_path):
            with open(output_path, "w", encoding='utf-8') as f:
                pass
    # {
    #     "input": "The input/command for the task, usually short, such as questions in QA, queries in Few-shot tasks, etc",
    #     "context": "The documents input into the long-text task.",
    #     "answers": "A List of all true answers",
    #     "length": "Total length of the first three items (counted in characters for Chinese and words for English)",
    #     "dataset": "The name of the dataset to which this piece of data belongs",
    #     "language": "The language of this piece of data",
    #     "answer_keywords": "The key words or sentences manually filtered from the answers",
    #     "confusing_facts": "This key represents confusing facts inserted to context to make the evaluation more challenging.",
    # }
    import re
    from langdetect import detect
    pattern = r'###\s+Passage\s+\d+|###\s+文章\s+\d+'
    remain_samples = args.max_samples
    def calculate_length(text):
        if detect(text) == "en":
            return len(text.split(" "))
        else:
            return len(text)
    def data_generator(dataset_paths: List[str]):
        for dataset_path in dataset_paths:
            with open(dataset_path, "r", encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    context = data["context"]
                    passages = re.split(pattern, context)
                    for passage in passages:
                        if len(passage) < 10:
                            continue
                        yield (passage, dataset_path)
    # data_path_num_dict = {
    #     dataset_path : 0 for dataset_path in dataset_path_list
    # }
    # data_path_len_dict = {
    #     dataset_path : [] for dataset_path in dataset_path_list
    # }
    # for passage, dataset_path in data_generator(dataset_path_list):
    #     data_path_num_dict[dataset_path] += 1
    #     data_path_len_dict[dataset_path].append(calculate_length(passage))
    # print(data_path_num_dict)
    # for data_path, data_path_len in data_path_len_dict.items():
    #     print(data_path, sum(data_path_len) / len(data_path_len), min(data_path_len), max(data_path_len))
    # exit(0)
    def batch_data_generator(data_generator, batch_size):
        nonlocal remain_samples
        idx = 0
        batch = []
        for data in data_generator:
            batch.append(data)
            idx += 1
            if remain_samples is not None:
                remain_samples -= 1
            if remain_samples is not None and remain_samples <= 0:
                yield batch
                batch = []
                break
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch
    async def get_lv_eval_data(passage, dataset_path):
        language = args.language
        id = dataset_path.split("/")[-1].split(".")[0]
        output_path = os.path.join(args.output_path, id + ".jsonl")
        GENERATE_EDIT_PROMPT = GENERATE_EDIT_PROMPT_EN if language == "en" else GENERATE_EDIT_PROMPT_ZH # type: ignore
        generate_lv_eval_prompt = GENERATE_EDIT_PROMPT.format( # type: ignore
            text = passage,
            metadata = "There is no metadata for this data, please generate the edit suggestion based on the text only." if language == "en" else "这个数据没有元数据，请根据文本生成编辑建议。"
        )
        response = await llm.get_response(generate_lv_eval_prompt, **llm_args)
        result = {
            "clean_text" : passage,
            "edit_suggestion" : response,
            "id" : id,
        }
        with open(output_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return
        
    total_len = 0
    for _ in data_generator(dataset_path_list):
        total_len += 1
    total_len = min(total_len, args.max_samples) if args.max_samples is not None else total_len
    for idx, batch_data in enumerate(batch_data_generator(data_generator(dataset_path_list), args.batch_size)):
        tasks = []
        for passage, dataset_path in batch_data:
            tasks.append(execute_with_retry_async(get_lv_eval_data, passage=passage, dataset_path=dataset_path))
        await tqdm_asyncio.gather(*tasks, desc="Process lv eval datas batch " + str(idx + 1) + "/" + str(math.ceil(total_len / args.batch_size)))
            

    args.max_samples = remain_samples

async def get_long_bench_datas(args, llm, dataset_path_list, output_path, max_length, **llm_args):
    assert len(dataset_path_list) == 1
    dataset_path = dataset_path_list[0]
    dataset_name_list = [
        # "gov_report_e",
        # "qasper" ,
        # "qmsum" ,
        # "musique" ,
        # "multifieldqa_en", 
        "multifieldqa_zh"
    ]
    dataset_paths = [
        os.path.join(dataset_path, dataset_name + ".jsonl")
        for dataset_name in dataset_name_list
    ]
    remain_samples = args.max_samples
    def data_generator(dataset_paths: List[str]):
        for dataset_path in dataset_paths:
            with open(dataset_path, "r", encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if dataset_path.endswith("gov_report_e.jsonl"):
                        yield (data["context"], dataset_path, {
                            "summary" : data['answers'],
                            "explanation" : "The summary of the text" if data['language'] == "en" else "文本摘要"
                        })
                    elif dataset_path.endswith("qasper.jsonl") or dataset_path.endswith("qmsum.jsonl") or dataset_path.endswith("multifieldqa_en.jsonl") or dataset_path.endswith("multifieldqa_zh.jsonl"):
                        yield (data["context"], dataset_path, {
                            "question" : data['input'],
                            "answer" : data['answers'],
                            "explanation" : "The question and answer of the text which can help you to understand the text better and generate a better edit suggestion." if data['language'] == "en" else "文本的问题和答案，这些数据可以帮助您更好地理解文本并生成更好的编辑建议。"
                        })
                    elif dataset_path.endswith("musique.jsonl"):
                        pattern = r'Passage\s+\d+'
                        passages = re.split(pattern, data["context"])
                        for passage in passages:
                            if len(passage) < 10:
                                continue
                            yield (passage, dataset_path, None)
                    
    
    
    def batch_data_generator(data_generator, batch_size):
        nonlocal remain_samples
        idx = 0
        batch = []
        for data in data_generator:
            batch.append(data)
            idx += 1
            if remain_samples is not None:
                remain_samples -= 1
            if remain_samples is not None and remain_samples <= 0:
                yield batch
                batch = []
                break
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch
    async def get_long_bench_data(passage, dataset_path, metadata):
        language = args.language
        id = dataset_path.split("/")[-1].split(".")[0]
        output_path = os.path.join(args.output_path, id + ".jsonl")
        generate_long_bench_prompt = GENERATE_EDIT_PROMPT.format( # type: ignore
            text = passage,
            metadata = "There is no metadata for this data, please generate the edit suggestion based on the text only." if language == "en" else "这个数据没有元数据，请根据文本生成编辑建议。"
        ) if metadata is None else GENERATE_EDIT_PROMPT.format( # type: ignore
            text = passage,
            metadata = str(metadata)
        )

        response = await llm.get_response(generate_long_bench_prompt, **llm_args)
        result = {
            "clean_text" : passage,
            "edit_suggestion" : response,
            "id" : id,
        }
        with open(output_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return
    total_len = 0
    for _ in data_generator(dataset_paths):
        total_len += 1
    total_len = min(total_len, args.max_samples) if args.max_samples is not None else total_len
    for idx, batch_data in enumerate(batch_data_generator(data_generator(dataset_paths), args.batch_size)):
        tasks = []
        for passage, dataset_path, metadata in batch_data:
            tasks.append(execute_with_retry_async(get_long_bench_data, passage=passage, dataset_path=dataset_path, metadata=metadata))
        await tqdm_asyncio.gather(*tasks, desc="Process long bench datas batch " + str(idx + 1) + "/" + str(math.ceil(total_len / args.batch_size)))
    

                        
                
    
def main(args):
    llm = AsyncLLM(args.chat_model, args.chat_base_url, args.chat_api_key, log_path=None)
    llm_args = {
        "max_tokens" : 1024,
        "is_stream" : True,
        "temperature" : 0,
        "log" : False
    }
    if os.path.isfile(args.output_path):
        # clear the file
        with open(args.output_path, "w", encoding='utf-8') as f:
            pass

    tasks = []
    for data_path in args.dataset_paths:
        if "narrativeqa" in data_path.lower():
            max_length = args.max_length if args.max_length != "auto" else 8192
            tasks.append(get_narrative_qa_datas(args, llm, [data_path], args.output_path, max_length, **llm_args))
        elif "quality" in data_path.lower():
            max_length = args.max_length if args.max_length != "auto" else None 
            tasks.append(get_quality_datas(args, llm, [data_path], args.output_path, max_length, **llm_args))
        elif "lveval" in data_path.lower():
            max_length = args.max_length if args.max_length != "auto" else None
            tasks.append(get_lv_eval_datas(args, llm, [data_path], args.output_path, max_length, **llm_args))
        elif "long_bench" in data_path.lower():
            max_length = args.max_length if args.max_length != "auto" else 8192
            tasks.append(get_long_bench_datas(args, llm, [data_path], args.output_path, max_length, **llm_args))
        else:
            raise ValueError("invalid dataset path")
    async def run_tasks():
        await asyncio.gather(*tasks)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_tasks())
    loop.close()
    
    

if __name__ == "__main__":
    # for quality data dataset_paths is a list of file paths eg. /data1/ytshi/modification/datas/quality_v1.0.1/QuALITY.v1.0.1.htmlstripped.dev /...
    # for narrativeqa data dataset_paths is one directory path eg. /data1/ytshi/modification/datas/narrativeqa  
    # for lm_eval data dataset_paths is one directory path eg. /data1/ytshi/LVEval
    # for long_bench data dataset_paths is one directory path eg. /data1/ytshi/long_bench
    parser = ArgumentParser()
    parser.add_argument("--chat_model", default=None)
    parser.add_argument("--chat_api_key", default=None)
    parser.add_argument("--chat_base_url", default=None)
    parser.add_argument("--dataset_paths", nargs='+', default=None)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--language", default="en", type=str, choices=["en", "zh"])
    parser.add_argument("--max_length", default=None, type=str, help="the max length of the input text, for english is the word count, for chinese is the character count, can be 'auto', default is None which means no truncation")
    parser.add_argument("--batch_size", default=8, type=int, help="the batch size for the data generation only for lveval dataset because the data is too large 50000 + lines")
    args = parser.parse_args()
    get_prompt(args.language)
    main(args)