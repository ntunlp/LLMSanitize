# test min-K-prob model contamination method
python main.py \
--method min-prob \
--local_port 6002 \
--local_port_2 6002 \
--model_name qwen \
--model_name_2 qwen \
--top_logprobs 5 \
--max_tokens 1 \
--echo \
--eval_data_name allenai/ai2_arc \
--eval_data_config_name "ARC-Challenge" \
--eval_set_key test \
--text_keys "question+choices+answerKey" \
--max_request_time 10 \
--num_proc 64 \
--output_dir ./output_dir/min_prob/qwen/arc