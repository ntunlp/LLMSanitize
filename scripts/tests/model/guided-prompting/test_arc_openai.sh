# test guided prompting model contamination method
echo "this method might require running vllm serving locally"
python main.py \
--eval_data_name allenai/ai2_arc \
--eval_data_config_name "ARC-Challenge" \
--eval_set_key test \
--text_key question \
--label_key answerKey \
--n_eval_data_points 1000 \
--num_proc 1 \
--method guided-prompting \
--openai_creds_key_file "openai_creds/openai_api_key.txt" \
--local_api_type "openai" \
--guided_prompting_task_type CLS \
--use_local_model
