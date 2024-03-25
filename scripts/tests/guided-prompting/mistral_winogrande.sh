# test guided prompting model contamination method
echo "this method might require running vllm serving locally"
python main.py \
--eval_data_name winogrande \
--eval_data_config_name winogrande_debiased \
--eval_set_key test \
--text_key sentence \
--label_key answer_token \
--method guided-prompting \
--num_proc 40 \
--local_port 8008 \
--guided_prompting_task_type FIM \
--use_local_model \
--n_eval_data_points 1000 \
--model_name "/home/fangkai/pretrained-models/Mistral-7B-v0.1"
