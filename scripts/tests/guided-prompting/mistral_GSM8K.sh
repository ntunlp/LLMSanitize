# test guided prompting model contamination method
echo "this method might require running vllm serving locally"
python main.py \
--eval_data_name gsm8k \
--eval_data_config_name "main" \
--eval_set_key test \
--method guided-prompting \
--text_key question \
--label_key answer \
--num_proc 20 \
--local_port 8000 \
--guided_prompting_task_type QA \
--use_local_model \
--n_eval_data_points 1000 \
--model_name "mistralai/Mistral-7B-v0.1"
