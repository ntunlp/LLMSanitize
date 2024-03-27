model_name=/home/fangkai/pretrained-models/Llama-2-7b-chat-hf
port=8008
api_type=openai # [openai, post]

datasets=(arc gsm8k hellaswag mmlu truthfulqa winogrande)
for name in $datasets;
do 
    sh scripts/tests/guided-prompting/test_$name.sh -p $port -m $model_name -t $api_type
done