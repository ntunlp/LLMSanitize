### Data contamination use cases

# test gpt-2 string matching
#python main.py --dataset_name Rowan/hellaswag --method gpt-2
# # test gpt-2 string matching
# python main.py --dataset_name Rowan/hellaswag --method gpt-2

# test gpt-3 string matching
#python main.py --dataset_name Rowan/hellaswag --method gpt-3
# # test gpt-3 string matching
# python main.py --dataset_name Rowan/hellaswag --method gpt-3

# test palm string matching
#python main.py --dataset_name Rowan/hellaswag --method palm
# # test palm string matching
# python main.py --dataset_name Rowan/hellaswag --method palm

# test gpt-4 string matching
#python main.py --dataset_name Rowan/hellaswag --method gpt-4
# # test gpt-4 string matching
# python main.py --dataset_name Rowan/hellaswag --method gpt-4

# test platypus emebddings similarity
#python main.py --dataset_name Rowan/hellaswag --method platypus
# # test platypus emebddings similarity
# python main.py --dataset_name Rowan/hellaswag --method platypus


### Model contamination use cases

# test guided prompting
echo "this method might require running vllm serving locally"
python main.py --dataset_name ag_news --method guided-prompting --text_key text --num_proc 80 --local_port 8000 \
--guided_prompting_task_type CLS --use_local_model --model_name "deepseek-ai/deepseek-llm-7b-chat"

# sharded likelihood
# python main.py --dataset_name google/boolq \
# --text_keys "question+passage" \
# --eval_set_key validation \
# --method sharded-likelihood \
# --sharded_likelihood_model gpt2-xl \
# --sharded_likelihood_context_len 1024 \
# --sharded_likelihood_stride 512 \
# --sharded_likelihood_num_shards 15 \
# --sharded_likelihood_permutations_per_shard 25 \
# --sharded_likelihood_max_examples 1000 \
# --log_file_path "sharded-likelihood-result.log"