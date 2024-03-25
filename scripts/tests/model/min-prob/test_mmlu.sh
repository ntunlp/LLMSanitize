# test min-K-prob model contamination method
python main.py \
--method min-prob \
--local_port 6000 \
--local_port_2 6000 \
--model_name mistral \
--model_name_2 mistral \
--top_logprobs 5 \
--max_tokens 1 \
--echo \
--eval_data_name cais/mmlu \
--eval_data_config_name all \
--eval_set_key test \
--text_keys "question+choices+answer" \
--max_request_time 10 \
--num_proc 64 \
--output_dir ./output_dir/min_prob/mistral/mmlu