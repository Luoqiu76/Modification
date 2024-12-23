INPUT_PATHS = (
    "/data1/ytshi/modification/processed_datas/musique.jsonl",
    "/data1/ytshi/modification/processed_datas/qasper.jsonl",
    "/data1/ytshi/modification/processed_datas/qmsum.jsonl"
)

python /data1/ytshi/modification/async_main.py \
    --chat_api_key "sk-6yr7BWUpTq1f26CHrAV5OJ0i90GpJtyqRVMEuXPue2B4lRkj" \
    --chat_base_url "https://xiaoai.plus/v1" \
    --chat_model "gpt-4o-mini" \
    --input_path "/data1/ytshi/modification/processed_datas/cmrc_mixup_256k.jsonl" \
    --work_dir "/data1/ytshi/modification/gpt-4o-mini/lveval" \
    --depth_limit 1 \
    --chunk_size 4096 \
    --language "en" \
    --chunk_overlap 0 \
    --enable_log \
    --enable_baseline \
    --enable_stream \
    --async_stage 2 \
    --max_concurrent 20 \
    --max_rate_limit 1
