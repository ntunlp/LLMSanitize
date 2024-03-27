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
--eval_data_name allenai/ai2_arc \
--eval_data_config_name "ARC-Challenge" \
--eval_set_key test \
--method guided-prompting \
--text_key question \
--label_key answerKey \
--num_proc 40 \
--local_port $port \
--guided_prompting_task_type CLS \
--use_local_model \
--n_eval_data_points 1000 \
--model_name $model_name \
--local_api_type $local_api_type
