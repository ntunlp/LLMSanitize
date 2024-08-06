# Get the options
while getopts ":m:" option; do
   case $option in
      m) # Enter closed_data name
         model_name=$OPTARG;;
   esac
done

echo "model name ", $model_name

# test min-K-prob closed_data contamination method
python main.py \
--eval_data_name winogrande \
--eval_data_config_name winogrande_debiased \
--eval_set_key test \
--text_keys sentence+option1+option2+answer \
--n_eval_data_points 100 \
--num_proc 0 \
--method sharded-likelihood \
--model_name $model_name \
--sharded_likelihood_context_len 1024 \
--sharded_likelihood_stride 512 \
--sharded_likelihood_num_shards 15 \
--sharded_likelihood_permutations_per_shard 25 \
--sharded_likelihood_mp_prawn
