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
--eval_data_name Rowan/hellaswag \
--eval_set_key validation \
--text_key "ctx" \
--label_key activity_label \
--n_eval_data_points 1000 \
--num_proc 40 \
--method guided-prompting \
--local_port $port \
--local_api_type $local_api_type \
--model_name $model_name \
--guided_prompting_task_type NLI \
--use_local_model