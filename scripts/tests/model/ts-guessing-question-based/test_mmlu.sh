# Get the options
while getopts ":p:m:" option; do
   case $option in
      p) # port number
         port=$OPTARG;;
      m) # Enter model name
         model_name=$OPTARG;;
   esac
done

echo "model name ", $model_name
echo "local port: ", $port

# test guided prompting model contamination method
python main.py \
--eval_data_name cais/mmlu \
--eval_data_config_name all \
--eval_set_key test \
--text_key question \
--label_key answer_text \
--n_eval_data_points -1 \
--num_proc 40 \
--method ts-guessing-question-based \
--local_port $port \
--model_name $model_name \
--guided_prompting_task_type QA \
--use_local_model
