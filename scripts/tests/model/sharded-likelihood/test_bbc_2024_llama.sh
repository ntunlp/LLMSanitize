llama="/home/fangkai/pretrained-models/Llama-2-7b-chat-hf"

python main.py \
--eval_data_name "RealTimeData/bbc_news_alltime" \
--eval_data_config_name "2024-02" \
--eval_set_key train \
--text_keys "description+section+content" \
--n_eval_data_points 100 \
--method sharded-likelihood \
--sharded_likelihood_model gpt2-xl \
--sharded_likelihood_context_len 1024 \
--sharded_likelihood_stride 512 \
--sharded_likelihood_num_shards 15 \
--sharded_likelihood_permutations_per_shard 25 \
--sharded_likelihood_max_examples 1000 \
--sharded_likelihood_mp_prawn \