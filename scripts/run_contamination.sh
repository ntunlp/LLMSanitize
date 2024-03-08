### Data contamination use case

# test gpt-4 string matching data contamination method
python main.py \
--dataset_name Rowan/hellaswag \
--method gpt-4


### Model contamination use cases

# test guided prompting model contamination method
echo "this method might require running vllm serving locally"
python main.py \
--dataset_name ag_news \
--method guided-prompting \
--text_key text -\
-num_proc 80 \
--local_port 6001 \
--guided_prompting_task_type CLS \
--use_local_model \
--model_name "/home/fangkai/pretrained-models/Mistral-7B-v0.1"
