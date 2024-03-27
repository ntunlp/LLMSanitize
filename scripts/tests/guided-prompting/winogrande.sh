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

# test guided prompting model contamination method
echo "this method might require running vllm serving locally"
python main.py \
--eval_data_name winogrande \
--eval_data_config_name winogrande_debiased \
--eval_set_key test \
--text_key sentence \
--label_key answer_token \
--method guided-prompting \
--num_proc 40 \
--local_port $port \
--guided_prompting_task_type FIM \
--use_local_model \
--n_eval_data_points 1000 \
--model_name $model_name \
--local_api_type $local_api_type
