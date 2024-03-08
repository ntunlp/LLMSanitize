"""
Contamination detection class for model contamination use cases: func(llm, data)
"""

import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Value
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from llmsanitize.configs.config import supported_methods, config
from llmsanitize.utils.method_utils import guided_prompt_process_fn, sharded_likelihood_main, guided_prompt_filter_fn
from llmsanitize.base_contamination_checker import BaseContaminationChecker
from llmsanitize.min_prob_model_contamination_checker import evaluate_data
from llmsanitize.llm import LLM
   
from functools import partial

class ModelContaminationChecker(BaseContaminationChecker):
    def __init__(self, args):
        super(ModelContaminationChecker, self).__init__(args)
        self.args = args

    def run_contamination(self, method):
        if not (method in self.supported_methods.keys()):
            methods = list(self.supported_methods.keys())
            raise KeyError(f'Please pass in a data contamination method which is supported, among: {methods}')

        if method == "guided-prompting":
            self.contamination_guided_prompting()

        if method == "sharded-likelihood":
            self.sharded_likelihood_comparison_test()

        if method == "min-prob":
            self.min_prob_comparison()

    def contamination_guided_prompting(self):
        import llmsanitize.prompts.guided_prompting.general_instructions as gi_prompts
        import llmsanitize.prompts.guided_prompting.guided_instructions as gui_prompts
        # method-specific dataset processing:
        ## only examine eval data here
        process_fn = guided_prompt_process_fn

        # based on task type, choose prompt template
        type_str = self.guided_prompting_task_type
        guided_template = getattr(gui_prompts, f"GUI_{type_str}")
        general_template = getattr(gi_prompts, f"GI_{type_str}")

        # process selected examples parallely
        num_examples_to_test = 800
        random_examples = self.eval_data.shuffle(seed=42).filter(partial(guided_prompt_filter_fn, text_key=self.text_key))\
                                                        .filter(lambda _, idx: idx < num_examples_to_test, with_indices=True)

        llm = LLM(local_port=self.local_port, model_name=self.model_name)
        process_fn = partial(process_fn, llm=llm,
                       split_name=self.eval_set_key, dataset_name=self.eval_data_name, label_key=self.label_key,
                       text_key=self.text_key, general_template=general_template, guided_template=guided_template)
        
        # somehow I need to do this to avoid datasets bug (https://github.com/huggingface/datasets/issues/6020#issuecomment-1632803184)
        features = self.eval_data.features
        features['general_score'] = Value(dtype='float64', id=None)
        features['guided_score'] = Value(dtype='float64', id=None)
        features["general_response"] = Value(dtype='string', id=None)
        features["guided_response"] = Value(dtype='string', id=None)
        features["first_part"] = Value(dtype='string', id=None)
        features["second_part"] = Value(dtype='string', id=None)
        
        processed_examples = random_examples.map(process_fn, with_indices=True, num_proc=self.num_proc, features=features)\
                            .filter(lambda example: len(example['general_response']) > 0 and len(example['guided_response']) > 0)
        
        scores_diff = [example['guided_score']-example['general_score'] for example in processed_examples]
        print(f"Tested {len(processed_examples)} examples with guided-prompting for model {self.model_name}")
        print(f"guided_score - general_score (RougeL)\nmean: {np.mean(scores_diff):.2f}, std: {np.std(scores_diff):.2f}")
        # TODO: add significance measure and bootstrap resampling
        print("skipping the bootstrap resampling and significance measure for now")

    def sharded_likelihood_comparison_test(self):
        sharded_likelihood_main(self.sharded_likelihood_model,
                                self.eval_data,
                                context_len=self.sharded_likelihood_context_len,
                                stride=self.sharded_likelihood_stride,
                                num_shards=self.sharded_likelihood_num_shards,
                                permutations_per_shard=self.sharded_likelihood_permutations_per_shard,
                                random_seed=self.seed,
                                max_examples=self.sharded_likelihood_max_examples,
                                log_file_path=self.log_file_path)

    def min_prob_comparison(self):
        """
        The command for testing:
        python main.py --method min-prob --local_port 6000 --local_port_2 6000 --model_name gemma-2b --model_name_2 gemma-2b --top_logprobs 2 \\
        --eval_data_name swj0419/WikiMIA --eval_set_key WikiMIA_length32 --text_key input --max_request_time 5
        """
        results = evaluate_data(self, self.eval_data)
        print(results)
