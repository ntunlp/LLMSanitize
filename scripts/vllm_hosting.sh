export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export RAY_memory_monitor_refresh_ms=0; export CUDA_VISIBLE_DEVICES=0,1,2,3; \
server_type=vllm.entrypoints.openai.api_server

python -m $server_type \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --disable-log-requests \
    --host 127.0.0.1 --tensor-parallel-size 4 \
    --download_dir /export/home/cache \