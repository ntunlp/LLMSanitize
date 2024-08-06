# Get the options
while getopts ":p:m:" option; do
   case $option in
      p) # port number
         port=$OPTARG;;
      m) # Enter closed_data name
         model_name=$OPTARG;;
   esac
done

echo "model name ", $model_name
echo "local port: ", $port

# test min-K-prob closed_data contamination method
python main.py \
--eval_data_name winogrande \
--eval_data_config_name winogrande_debiased \
--eval_set_key test \
--text_keys sentence+option1+option2+answer \
--n_eval_data_points 100 \
--num_proc 0 \
--method min-prob \
--local_port $port \
--model_name $model_name \
--max_output_tokens 1 \
--top_logprobs 5 \
--max_request_time 10 \
--echo \
--minkprob_local_port_2 $port \
--minkprob_model_name_2 $model_name
