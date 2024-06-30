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
--eval_data_name gsm8k \
--eval_data_config_name main \
--eval_set_key test \
--text_key question \
--label_key answer \
--n_eval_data_points 100 \
--method cdd \
--model_name $model_name \
--local_port $port \
--num_samples 20 \
--temperature 0.8 \
--cdd_alpha 0.05 \
--cdd_xi 0.01