# test min-K-prob model contamination method
python main.py \
--method min-prob \
--local_port 6000 \
--local_port_2 6000 \
--model_name llama \
--model_name_2 llama \
--top_logprobs 5 \
--max_tokens 1 \
--echo \
--eval_data_name truthful_qa \
--eval_data_config_name generation \
--eval_set_key "validation" \
--text_keys "question+best_answer" \
--max_request_time 10 \
--num_proc 64 \
--output_dir ./output_dir/min_prob/llama2/truthful_qa