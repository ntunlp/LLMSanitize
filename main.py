from llmsanitize import DataContaminationChecker, ModelContaminationChecker
from llmsanitize.configs.config import supported_methods
from llmsanitize.utils.utils import seed_everything
import multiprocessing as mp
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--dataset_name", type=str, default="")  # ["Rowan/hellaswag"]
    parser.add_argument("--train_data_name", type=str, default="", help="training dataset name")  # ["Rowan/hellaswag"]
    parser.add_argument("--eval_data_name", type=str, default="", help="eval dataset name")  # ["Rowan/hellaswag"]
    parser.add_argument("--eval_set_key", type=str, default="test", help="eval set key")
    parser.add_argument("--text_key", type=str, default="ctx", help="the key to text content of each data instance.")
    parser.add_argument("--text_keys", type=str, default="", help="the keys of text contents to be combined of each data instance - pass them as key_1+key_2.")
    parser.add_argument("--label_key", type=str, default="label", help="the key to label content of each data instance.")
    # parser.add_argument("--use_local_model", action='store_true', default=False)
    parser.add_argument("--local_model_path", default=None, help="local model path for non-service based inference.")
    parser.add_argument("--local_tokenizer_path", default=None, help="local tokenizer path for non-service based inference.")
    parser.add_argument("--num_proc", type=int, default=20, help="recommend: 20 for openai calls, 80 for local calls")
    parser.add_argument("--method_name", type=str, choices=supported_methods.keys())
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--log_file_path", type=str, default="log.txt", help="log file path")
    # OpenAI API or vLLM inference arguments
    parser.add_argument("--openai_creds_key_file", type=str, default=None, help="OpenAI API key file path.")
    parser.add_argument("--openai_creds_key_file_2", type=str, default=None, help="OpenAI API key file path.")
    parser.add_argument("--local_port", type=str, default=None, help="Local model port for service based inference.")
    parser.add_argument("--local_port_2", type=str, default=None, help="Local model port for service based inference.")  # TODO: If there is better way to initialize two models.
    parser.add_argument("--model_name", type=str, default=None, help="model name for service based inference.")
    parser.add_argument("--model_name_2", type=str, default=None, help="model name for service based inference.")
    parser.add_argument("--num_samples", type=int, default=1, help="number of samples to generate")
    parser.add_argument("--max_tokens", type=int, default=128, help="max tokens for each sample")
    parser.add_argument("--top_logprobs", type=int, default=0, help="top logprobs for each sample")
    parser.add_argument("--max_request_time", type=int, default=0, help="max request time for each sample")
    parser.add_argument("--sleep_time", type=int, default=0, help="sleep time for each sample")
    # Method specific-arguments
    ### Guided prompting
    parser.add_argument("--guided_prompting_task_type", choices=["CLS", "NLI", "SUM", "XSUM"],
                        help="For guided-prompting: set task type to either {classification, NLI, summarization, extreme-summarization}")
    parser.add_argument("--use_local_model", action='store_true', default=False)
    ### Sharded likelihood
    parser.add_argument("--sharded_likelihood_model", type=str, default="gpt2-xl", help="For sharded-likelihood: set model name or path")
    parser.add_argument("--sharded_likelihood_context_len", type=int, default=1024, help="For sharded-likelihood: set context length")
    parser.add_argument("--sharded_likelihood_stride", type=int, default=512, help="For sharded-likelihood: set stride length")
    parser.add_argument("--sharded_likelihood_num_shards", type=int, default=15, help="For sharded-likelihood: set number of shards")
    parser.add_argument("--sharded_likelihood_permutations_per_shard", type=int, default=25,
                        help="For sharded-likelihood: set number of permutations per shard")
    parser.add_argument("--sharded_likelihood_max_examples", type=int, default=5000, help="For sharded-likelihood: set max examples")
    parser.add_argument("--sharded_likelihood_mp_prawn", action='store_true', default=False)
    args = parser.parse_args()
    # if dataset name is set, set train_set and eval_set to dataset_name
    if len(args.dataset_name) > 0:
        args.train_data_name = args.dataset_name
        args.eval_data_name = args.dataset_name
    args.text_keys = args.text_keys.split("+")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)

    check_args(args)

    if args.sharded_likelihood_mp_prawn:
        mp.set_start_method('spawn')

    # assign data / model contamination checker based on method type
    assert args.method_name in supported_methods, f"Error, {args.method_name} not in supported methods: {list(supported_methods.keys())}"
    if supported_methods[args.method_name]['type'] == 'data':
        ContaminationChecker = DataContaminationChecker
    elif supported_methods[args.method_name]['type'] == 'model':
        ContaminationChecker = ModelContaminationChecker

    contamination_checker = ContaminationChecker(args)
    contamination_checker.run_contamination(args.method_name)


def check_args(args):
    assert args.method_name in supported_methods, f"Error, {args.method_name} not in supported methods: {list(supported_methods.keys())}"
    assert args.text_key != "" or args.text_key != [], f"Error, specify some text key"


if __name__ == '__main__':
    main()
