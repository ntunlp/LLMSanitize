from lmsanitize import DataContaminationChecker
from lmsanitize.configs.config import supported_methods
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="") # ["Rowan/hellaswag"]
    parser.add_argument("--train_data_name", type=str, default="", help="training dataset name") # ["Rowan/hellaswag"]
    parser.add_argument("--eval_data_name", type=str, default="", help="eval dataset name") # ["Rowan/hellaswag"]
    parser.add_argument("--eval_set_key", type=str, default="test", help="eval set key")
    parser.add_argument("--text_key", type=str, default="ctx", help="the key to text content of each data instance.")
    parser.add_argument("--label_key", type=str, default="label", help="the key to label content of each data instance.")
    parser.add_argument("--use_local_model", action='store_true')
    parser.add_argument("--num_proc", type=int, default=20, help="recommend: 20 for openai calls, 80 for local calls")
    parser.add_argument("--method_name", type=str, choices=supported_methods.keys())
    args = parser.parse_args()
    # if dataset name is set, set train_set and eval_set to dataset_name
    if len(args.dataset_name) > 0:
        args.train_data_name = args.dataset_name
        args.eval_data_name = args.dataset_name
    return args

def main():
    args = parse_args()
    contamination_checker = DataContaminationChecker(args)
    contamination_checker.run_contamination(args.method_name)

if __name__ == '__main__':
    main()