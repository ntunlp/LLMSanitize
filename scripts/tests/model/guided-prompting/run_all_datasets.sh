model_name="<path_to_your_local_model>"
port=8008
api_type=openai # [openai, post]

datasets=(arc gsm8k hellaswag mmlu truthfulqa winogrande)
for name in $datasets;
do 
    sh scripts/tests/guided-prompting/test_$name.sh -p $port -m $model_name -t $api_type
done