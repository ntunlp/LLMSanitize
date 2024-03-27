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
            eval_data=self.eval_data,
            model_name=self.model_name,
            context_len=self.sharded_likelihood_context_len,
            stride=self.sharded_likelihood_stride,
            num_shards=self.sharded_likelihood_num_shards,
            permutations_per_shard=self.sharded_likelihood_permutations_per_shard,
            log_file_path=self.log_file_path
        )

    def contamination_min_prob(self):
        main_min_prob(
            test_data=self.eval_data,
            openai_creds_key_file=self.openai_creds_key_file,
            openai_creds_key_file_2=self.openai_creds_key_file_2,
            local_port=self.local_port,
            local_port_2=self.local_port_2,
            local_model_path=self.local_model_path,
            # local_model_path_2=self.local_model_path_2,  # Currently it is not contained in argument list. Uncomment this when you need it.
            local_tokenizer_path=self.local_tokenizer_path,
            # local_tokenizer_path_2=self.local_tokenizer_path_2,
            model_name=self.model_name,
            model_name_2=self.model_name_2,
            num_samples=self.num_samples,
            max_tokens=self.max_tokens,
            top_logprobs=self.top_logprobs,
            max_request_time=self.max_request_time,
            sleep_time=self.sleep_time,
            echo=self.echo,
            num_proc=self.num_proc,
            output_dir=self.output_dir,
            do_infer=self.do_infer,
        )
