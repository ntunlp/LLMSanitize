n_points=100000
method="platypus"

# ARC-Challenge / validation
python main.py \
--dataset_name allenai/ai2_arc \
--train_data_config_name "ARC-Challenge" \
--eval_set_key validation \
--text_keys "question+choices+answerKey" \
--n_eval_data_points $n_points \
--method $method

# ARC-Challenge / test
python main.py \
--dataset_name allenai/ai2_arc \
--train_data_config_name "ARC-Challenge" \
--eval_set_key test \
--text_keys "question+choices+answerKey" \
--n_eval_data_points $n_points \
--method $method

# HellaSwag / validation
python main.py \
--dataset_name Rowan/hellaswag \
--eval_set_key validation \
--text_keys "ctx+endings" \
--n_eval_data_points $n_points \
--method $method

# HellaSwag / test
python main.py \
--dataset_name Rowan/hellaswag \
--eval_set_key test \
--text_keys "ctx+endings" \
--n_eval_data_points $n_points \
--method $method

# Winogrande / validation
python main.py \
--dataset_name winogrande \
--train_data_config_name "winogrande_debiased" \
--eval_set_key validation \
--text_keys "sentence+option1+option2+answer" \
--n_eval_data_points $n_points \
--method $method

# Winogrande / test
python main.py \
--dataset_name winogrande \
--train_data_config_name "winogrande_debiased" \
--eval_set_key test \
--text_keys "sentence+option1+option2+answer" \
--n_eval_data_points $n_points \
--method $method

# GSM8K / test
python main.py \
--dataset_name gsm8k \
--train_data_config_name "main" \
--eval_set_key test \
--text_keys "question+answer" \
--n_eval_data_points $n_points \
--method $method