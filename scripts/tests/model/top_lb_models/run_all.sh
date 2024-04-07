# Whenever you change model:
# Select a model in start_vllm.sh
# Run start_vllm.sh
# Select the same model in this file
# Run this file (in another tab)

model=/home/fangkai/pretrained-models/Llama-2-7b-chat-hf # DEBUGGING
#model="davidkim205/Rhea-72b-v0.5" # n.1
#model="MTSAIR/MultiVerse_70B" # n.2 and 3
#model="SF-Foundation/Ein-72B-v0.11" # n.4
#model="SF-Foundation/Ein-72B-v0.13" # n.5
#model="SF-Foundation/Ein-72B-v0.12" # n.6
#model="abacusai/Smaug-72B-v0.1" # n.7
#model="ibivibiv/alpaca-dragon-72b-v1" # n.8
#model="moreh/MoMo-72B-lora-1.8.7-DPO" # n.9
#model="cloudyu/TomGrc_FusionNet_34Bx2_MoE_v0.1_DPO_f16" # n.10
#model="saltlux/luxia-21.4b-alignment-v1.0" # n.11
port=6000
api_type=post # [openai, post]

datasets=(arc gsm8k hellaswag mmlu truthfulqa winogrande)
vllm_methods=(guided-prompting min-prob cdd)
other_methods=(sharded-likelihood)

for name in $datasets;
do
    # methods requiring the vLLM: {guided-prompting, min-k prob, cdd}
    for method in $vllm_methods;
    do
        sh scripts/tests/$method/test_$name.sh -m $model -p $port -t $api_type
    done

    # method not requiring the vLLM: {sharded-likelihood}
    for method in $other_methods;
    do
        sh scripts/tests/$method/test_$name.sh -m $model
    done
done

