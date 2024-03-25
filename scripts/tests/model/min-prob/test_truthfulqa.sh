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
--eval_data_name truthful_qa \
--eval_data_config_name generation \
--eval_set_key "validation" \
--text_keys "question" \
--max_request_time 10 \
--num_proc 0 \
--output_dir ./output_dir/min_prob/misral/truthful_qa