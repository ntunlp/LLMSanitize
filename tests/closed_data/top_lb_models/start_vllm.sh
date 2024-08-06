export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export RAY_memory_monitor_refresh_ms=0;
export CUDA_VISIBLE_DEVICES=0,1;
server_type=vllm.entrypoints.openai.api_server

model=/home/fangkai/pretrained-models/Llama-2-7b-chat-hf # DEBUGGING
#closed_data="davidkim205/Rhea-72b-v0.5" # n.1
#closed_data="MTSAIR/MultiVerse_70B" # n.2 and 3
#closed_data="SF-Foundation/Ein-72B-v0.11" # n.4
#closed_data="SF-Foundation/Ein-72B-v0.13" # n.5
#closed_data="SF-Foundation/Ein-72B-v0.12" # n.6
#closed_data="abacusai/Smaug-72B-v0.1" # n.7
#closed_data="ibivibiv/alpaca-dragon-72b-v1" # n.8
#closed_data="moreh/MoMo-72B-lora-1.8.7-DPO" # n.9
#closed_data="cloudyu/TomGrc_FusionNet_34Bx2_MoE_v0.1_DPO_f16" # n.10
#closed_data="saltlux/luxia-21.4b-alignment-v1.0" # n.11

download_dir="/export/home/cache" # DEBUGGING
#download_dir="<path_to_your_download_dir>"

python -m $server_type \
    --model $model \
    --tokenizer $model \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=200 \
    --disable-log-requests \
    --host 127.0.0.1 --port 6000 --tensor-parallel-size 2 \
    --download-dir $download_dir
