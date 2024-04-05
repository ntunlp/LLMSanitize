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
python main.py \
--eval_data_name allenai/ai2_arc \
--eval_data_config_name "ARC-Challenge" \
--eval_set_key test \
--text_key question \
--label_key answerKey \
--n_eval_data_points 100 \
--num_proc 40 \
--method guided-prompting \
--local_port $port \
--local_api_type $local_api_type \
--model_name $model_name \
--guided_prompting_task_type CLS \
--use_local_model
