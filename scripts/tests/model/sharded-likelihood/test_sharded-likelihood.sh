# test sharded-likelihood model contamination method
python main.py \
--dataset_name google/boolq \
--text_keys question+answer+passage \
--eval_set_key validation \
--method sharded-likelihood \
--model_name gpt2-xl \
--sharded_likelihood_context_len 1024 \
--sharded_likelihood_stride 512 \
--sharded_likelihood_num_shards 15 \
--sharded_likelihood_permutations_per_shard 25 \
--sharded_likelihood_mp_prawn \
--log_file_path "sharded-likelihood-result.log"
