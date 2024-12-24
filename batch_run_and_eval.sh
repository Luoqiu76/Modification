#!/bin/bash

# api config for openai api
CHAT_API_KEY="sk-6yr7BWUpTq1f26CHrAV5OJ0i90GpJtyqRVMEuXPue2B4lRkj"
CHAT_BASE_URL="https://xiaoai.plus/v1"
CHAT_MODEL="gpt-4o-mini"
JUDGE_MODEL="gpt-4o-mini"


# azure config run model and judge model
export CONFIG_NAME=""
export JUDGE_CONFIG_NAME=""



# max concurrent coroutines in every time
MAX_CONCURRENT=4
# wait time when one coroutine is finished 
MAX_RATE_LIMIT=2
# batch size for api request
BATCH_SIZE=128


# pipline config
DEPTH_LIMIT=1
CHUNK_SIZE=4096
CHUNK_OVERLAP=0





declare -A LANGUAGE_INPUT_PATHS

LANGUAGE_INPUT_PATHS["en"]="
./processed_datas/data_narrative_qa.jsonl \
./processed_datas/data_quality_v1.0.1_train_dev_test.jsonl \
./processed_datas/gov_report_e.jsonl \
./processed_datas/multifieldqa_en.jsonl \
./processed_datas/musique.jsonl \
./processed_datas/qasper.jsonl \
./processed_datas/qmsum.jsonl
"


LANGUAGE_INPUT_PATHS["zh"]="
./processed_datas/multifieldqa_zh.jsonl
"  


SAVE_DIR="./results/$CHAT_MODEL"


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
        
        
        python async_main.py \
            --chat_api_key "$CHAT_API_KEY" \
            --chat_base_url "$CHAT_BASE_URL" \
            --chat_model "$CHAT_MODEL" \
            --input_path "$INPUT_PATH" \
            --work_dir "$WORK_DIR" \
            --depth_limit "$DEPTH_LIMIT" \
            --chunk_size "$CHUNK_SIZE" \
            --language "$LANGUAGE" \
            --chunk_overlap "$CHUNK_OVERLAP" \
            --enable_log \
            --enable_baseline \
            --enable_stream \
            --async_stage 2 \
            --max_concurrent "$MAX_CONCURRENT" \
            --max_rate_limit "$MAX_RATE_LIMIT" \
            --batch_size "$BATCH_SIZE" \
            # --enable_azure \
        
        echo "Start eval"
        
       
        python eval.py \
            --chat_api_key "$CHAT_API_KEY" \
            --chat_base_url "$CHAT_BASE_URL" \
            --chat_model "$CHAT_MODEL" \
            --work_dir "$WORK_DIR" \
            --enable_async \
            --enable_stream \
            --language "$LANGUAGE" \
            --max_concurrent "$MAX_CONCURRENT" \
            --max_rate_limit "$MAX_RATE_LIMIT" \
            --batch_size "$BATCH_SIZE" \
            # --enable_azure \
    done
done
