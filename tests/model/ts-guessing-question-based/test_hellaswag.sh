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
--eval_data_name Rowan/hellaswag \
--eval_set_key validation \
--text_key ctx \
--label_key activity_label \
--n_eval_data_points 100 \
--method ts-guessing-question-based \
--local_port $port \
--model_name $model_name \
#--ts_guessing_type_hint \
#--ts_guessing_category_hint \
#--ts_guessing_url_hint \
