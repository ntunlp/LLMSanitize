# test gpt-2 string matching
python main.py --dataset_name Rowan/hellaswag --method gpt-2

# test gpt-3 string matching
python main.py --dataset_name Rowan/hellaswag --method gpt-3

# test palm string matching
#python main.py --dataset_name Rowan/hellaswag --method palm

# test gpt-4 string matching
python main.py --dataset_name Rowan/hellaswag --method gpt-4

# test guided prompting
#echo "this method might require running vllm serving locally"
#python main.py --dataset_name ag_news --method guided-prompting --text_key text --guided_prompting_task_type CLS --use_local_model
