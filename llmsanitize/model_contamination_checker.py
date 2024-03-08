"""
Contamination detection class for model contamination use cases: func(llm, data)
"""

from llmsanitize.base_contamination_checker import BaseContaminationChecker
from llmsanitize.model_methods.guided_prompting import main_guided_prompting
from llmsanitize.model_methods.sharded_likelihood import main_sharded_likelihood
from llmsanitize.model_methods.min_prob import main_min_prob


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
            self.contamination_sharded_likelihood()
        if method == "min-prob":
            self.contamination_min_prob()

    def contamination_guided_prompting(self):
        main_guided_prompting(
            guided_prompting_task_type=self.guided_prompting_task_type,
            eval_data=self.eval_data,
            eval_data_name=self.eval_data_name,
            eval_set_key=self.eval_set_key,
            text_key=self.text_key,
            label_key=self.label_key,
            local_port=self.local_port,
            model_name=self.model_name,
            num_proc=self.num_proc
        )

    def contamination_sharded_likelihood(self):
        main_sharded_likelihood(
            model_name_or_path=self.sharded_likelihood_model,
            dataset_path=self.eval_data,
            context_len=self.sharded_likelihood_context_len,
            stride=self.sharded_likelihood_stride,
            num_shards=self.sharded_likelihood_num_shards,
            permutations_per_shard=self.sharded_likelihood_permutations_per_shard,
            random_seed=self.seed,
            max_examples=self.sharded_likelihood_max_examples,
            log_file_path=self.log_file_path
        )

    def contamination_min_prob(self):
        results = main_min_prob(
            args=self,
            test_data=self.eval_data
        )
        print(results)
