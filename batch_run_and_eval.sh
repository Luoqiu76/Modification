#!/bin/bash
# ./processed_datas/data_narrative_qa.jsonl \
# ./processed_datas/data_quality_v1.0.1_train_dev_test.jsonl \
# ./processed_datas/gov_report_e.jsonl \
# ./processed_datas/multifieldqa_en.jsonl \
# ./processed_datas/musique.jsonl \
# ./processed_datas/qasper.jsonl \
# ./processed_datas/qmsum.jsonl


# ./processed_datas/multifieldqa_zh.jsonl
# api config for openai api
# CHAT_API_KEY="sk-6yr7BWUpTq1f26CHrAV5OJ0i90GpJtyqRVMEuXPue2B4lRkj"
# CHAT_BASE_URL="https://xiaoai.plus/v1"
# CHAT_MODEL="gpt-4o-mini"
# JUDGE_MODEL="gpt-4o-mini"


# export OPENAI_API_KEY=""
# export CONFIG_NAME="gpt-4o"
# export JUDGE_CONFIG_NAME="gpt-4o"



# # max concurrent coroutines in every time
# MAX_CONCURRENT=64
# # wait time when one coroutine is finished 
# MAX_RATE_LIMIT=2
# # batch size for api request
# BATCH_SIZE=128


# pipline config
# DEPTH_LIMIT=1
# CHUNK_SIZE=4096
# CHUNK_OVERLAP=0
CHAT_MODEL="gpt-4o-mini"
RUN_NAME="chunk_size_no_limit"




declare -A LANGUAGE_INPUT_PATHS

LANGUAGE_INPUT_PATHS["en"]=" \
./processed_datas/musique.jsonl \
./processed_datas/qasper.jsonl \
./processed_datas/qmsum.jsonl \
./processed_datas/data_quality_v1.0.1_train_dev_test.jsonl \
"


LANGUAGE_INPUT_PATHS["zh"]="
./processed_datas/multifieldqa_zh.jsonl
"  


SAVE_DIR="./results/$CHAT_MODEL/$RUN_NAME"


for LANGUAGE in "en" "zh"; do
    echo "Processing language: $LANGUAGE"


    INPUT_PATHS="${LANGUAGE_INPUT_PATHS[$LANGUAGE]}"


    if [ -z "$INPUT_PATHS" ]; then
        echo "No input paths for $LANGUAGE, skipping..."
        continue
    fi


    for INPUT_PATH in $INPUT_PATHS; do
        echo "Processing file: $INPUT_PATH"
        

        FILENAME=$(basename "$INPUT_PATH" .jsonl)
        
       
        WORK_DIR="$SAVE_DIR/$FILENAME"
        echo "Working directory: $WORK_DIR"
        echo "Start run"
        
        
        mkdir -p "$WORK_DIR"
        
        
        python -u main.py \
        --input_path "$INPUT_PATH" \
        --work_dir "$WORK_DIR" \
        --language "$LANGUAGE"




        
        echo "Start eval"
        
       
        python -u eval.py \
            --work_dir "$WORK_DIR" \
            --language "$LANGUAGE"
    done
done

echo "Start summary"

python -u post_run.py \
--work_dir "$SAVE_DIR"