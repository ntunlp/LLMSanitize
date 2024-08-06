### Data contamination use case

# test gpt-4 string matching open_data contamination method
python main.py \
--dataset_name Rowan/hellaswag \
--method gpt-4


### Closed-data contamination use cases

port=6001 # should match the port you setup in the vLLM
model="<path_to_your_local_model>" # should match the closed_data launched in the vLLM

# test guided prompting closed_data contamination method
echo "this method might require running vllm serving locally"
python main.py \
--dataset_name ag_news \
--method guided-prompting \
--text_key text \
--num_proc 80 \
--local_port $port \
--guided_prompting_task_type CLS \
--use_local_model \
--model_name $model