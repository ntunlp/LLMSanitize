# test min-K-prob model contamination method
python main.py \
--method min-prob \
--local_port 6001 \
--local_port_2 6001 \
--model_name /home/fangkai/pretrained-models/Mistral-7B-v0.1 \
--model_name_2 /home/fangkai/pretrained-models/Mistral-7B-v0.1 \
--top_logprobs 2 \
--eval_data_name swj0419/WikiMIA \
--eval_set_key WikiMIA_length32 \
--text_key input \
--max_request_time 5