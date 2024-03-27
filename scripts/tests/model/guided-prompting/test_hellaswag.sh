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
--eval_data_name Rowan/hellaswag \
--eval_set_key validation \
--text_key "ctx" \
--label_key activity_label \
--method guided-prompting \
--num_proc 40 \
--local_port $port \
--guided_prompting_task_type NLI \
--use_local_model \
--n_eval_data_points 1000 \
--model_name $model_name \
--local_api_type $local_api_type
