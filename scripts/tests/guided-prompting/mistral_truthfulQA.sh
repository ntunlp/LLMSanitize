# test guided prompting model contamination method
echo "this method might require running vllm serving locally"
python main.py \
--eval_data_name truthful_qa \
--eval_data_config_name generation \
--eval_set_key validation \
--text_key question \
--label_key category \
--method guided-prompting \
--num_proc 40 \
--local_port 8008 \
--guided_prompting_task_type QA \
--use_local_model \
--n_eval_data_points 1000 \
--model_name "/home/fangkai/pretrained-models/Mistral-7B-v0.1"
