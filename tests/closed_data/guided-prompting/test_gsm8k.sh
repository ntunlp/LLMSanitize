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

# test guided prompting closed_data contamination method
echo "this method might require running vllm serving locally"
python main.py \
--eval_data_name gsm8k \
--eval_data_config_name main \
--eval_set_key test \
--text_key question \
--label_key answer \
--n_eval_data_points 100 \
--num_proc 40 \
--method guided-prompting \
--local_port $port \
--model_name $model_name \
--guided_prompting_task_type QA \
--use_local_model