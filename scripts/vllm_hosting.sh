# needed for the following methods:
# 1/ guided-prompting
# 2/ min-prob
# Run this script in one tab first, then run the script calling the method in another tab
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export RAY_memory_monitor_refresh_ms=0;
export CUDA_VISIBLE_DEVICES=2,3,4,5;
server_type=vllm.entrypoints.openai.api_server

python -m $server_type \
    --model /home/fangkai/pretrained-models/Qwen1.5-7B-Chat \
    --tokenizer /home/fangkai/pretrained-models/Qwen1.5-7B-Chat \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --disable-log-requests \
    --host 127.0.0.1 --port 6000 --tensor-parallel-size 4 \
    --download-dir /export/home/cache 
