# test guided prompting model contamination method
echo "this method might require running vllm serving locally"
python main.py \
--eval_data_name Rowan/hellaswag \
--eval_set_key validation \
--text_key "ctx" \
--label_key activity_label \
--method guided-prompting \
--num_proc 40 \
--local_port 8000 \
--guided_prompting_task_type NLI \
--use_local_model \
--n_eval_data_points 1000 \
--model_name "mistralai/Mistral-7B-v0.1"
