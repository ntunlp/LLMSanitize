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
python main.py \
--eval_data_name allenai/ai2_arc \
--eval_data_config_name ARC-Challenge \
--eval_set_key test \
--text_key question \
--label_key answerKey \
--n_eval_data_points 100 \
--method ts-guessing-question-multichoice \
--local_port $port \
--model_name $model_name