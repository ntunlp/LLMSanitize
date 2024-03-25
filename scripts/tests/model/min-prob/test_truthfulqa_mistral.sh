# test min-K-prob model contamination method
python main.py \
--method min-prob \
--local_port 6001 \
--local_port_2 6001 \
--model_name mistral-ins \
--model_name_2 mistral-ins \
--top_logprobs 5 \
--max_tokens 1 \
--echo \
--eval_data_name truthful_qa \
--eval_data_config_name generation \
--eval_set_key "validation" \
--text_keys "question+best_answer" \
--max_request_time 10 \
--num_proc 64 \
--output_dir ./output_dir/min_prob/mistral-ins/truthful_qa