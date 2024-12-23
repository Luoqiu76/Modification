


# qwen api key: sk-f748bf60646e4e3099cad81c46ef756c
# xiaoai api key: sk-tT0H1Lf77vnAhpvf981500F661B14f6eB350576eD02e5224
# qwen base url: https://dashscope.aliyuncs.com/compatible-mode/v1
# xiaoai base url: https://xiaoai.plus/v1
# model name qwen-max 
python /data1/ytshi/modification/eval.py \
    --chat_api_key "sk-6yr7BWUpTq1f26CHrAV5OJ0i90GpJtyqRVMEuXPue2B4lRkj" \
    --chat_base_url "https://xiaoai.plus/v1" \
    --chat_model "gpt-4o-mini" \
    --work_dir "/data1/ytshi/modification/gpt-4o-mini/lveval" \
    --enable_async \
    --enable_stream \
    --max_concurrent 20
