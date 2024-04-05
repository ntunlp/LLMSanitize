# Get the options
while getopts ":p:m:t:" option; do
   case $option in
      p) # port number
         port=$OPTARG;;
      m) # Enter model name
         model_name=$OPTARG;;
      t) # local api type
         local_api_type=$OPTARG;;
   esac
done

echo "model name ", $model_name
echo "local port: ", $port

# test min-K-prob model contamination method
python main.py \
--eval_data_name gsm8k \
--eval_data_config_name "main" \
--eval_set_key test \
--text_keys "question+answer" \
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
