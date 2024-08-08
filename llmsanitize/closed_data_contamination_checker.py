"""
Contamination detection class for closed_data contamination use cases: func(llm, open_data)
"""

from llmsanitize.base_contamination_checker import BaseContaminationChecker
from llmsanitize.closed_data_methods.guided_prompting import main_guided_prompting
from llmsanitize.closed_data_methods.sharded_likelihood import main_sharded_likelihood
from llmsanitize.closed_data_methods.min_prob import main_min_prob
from llmsanitize.closed_data_methods.cdd import main_cdd
from llmsanitize.closed_data_methods.ts_guessing_question_based import main_ts_guessing_question_based
from llmsanitize.closed_data_methods.ts_guessing_question_multichoice import main_ts_guessing_question_multichoice


class ClosedDataContaminationChecker(BaseContaminationChecker):
    def __init__(self, args):
        super(ClosedDataContaminationChecker, self).__init__(args)
        self.args = args

    def run_contamination(self, method):
        if not (method in self.supported_methods.keys()):
            methods = list(self.supported_methods.keys())
            raise KeyError(f'Please pass in a open_data contamination method which is supported, among: {methods}')

        if method == "guided-prompting":
            self.contamination_guided_prompting()
        elif method == "sharded-likelihood":
            self.contamination_sharded_likelihood()
        elif method == "min-prob":
            self.contamination_min_prob()
        elif method == "cdd":
            self.contamination_cdd()
        elif method == "ts-guessing-question-based":
            self.contamination_ts_guessing_question_based()
        elif method == "ts-guessing-question-multichoice":
            self.contamination_ts_guessing_question_multichoice()

    # to use with a vLLM instance
    def contamination_guided_prompting(self):
        main_guided_prompting(
            eval_data=self.eval_data,
            eval_data_name=self.eval_data_name,
            eval_set_key=self.eval_set_key,
            text_key=self.text_key,
            label_key=self.label_key,
            num_proc=self.num_proc,
            # closed_data parameters
            local_model_path=self.local_model_path,
            local_tokenizer_path=self.local_tokenizer_path,
            model_name=self.model_name,
            openai_creds_key_file=self.openai_creds_key_file,
            local_port=self.local_port,
            local_api_type=self.local_api_type,
            no_chat_template=self.no_chat_template,
            num_samples=self.num_samples,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_logprobs=self.top_logprobs,
            max_request_time=self.max_request_time,
            sleep_time=self.sleep_time,
            echo=self.echo,
            # method-specific parameters
            guided_prompting_task_type=self.guided_prompting_task_type,
        )

    def contamination_sharded_likelihood(self):
        main_sharded_likelihood(
            eval_data=self.eval_data,
            log_file_path=self.log_file_path,
            # closed_data parameters
            model_name=self.model_name,
            # method-specific parameters
            context_len=self.sharded_likelihood_context_len,
            stride=self.sharded_likelihood_stride,
            num_shards=self.sharded_likelihood_num_shards,
            permutations_per_shard=self.sharded_likelihood_permutations_per_shard,
        )

    # to use with a vLLM instance
    def contamination_min_prob(self):
        main_min_prob(
            eval_data=self.eval_data,
            num_proc=self.num_proc,
            output_dir=self.output_dir,
            # closed_data parameters
            local_model_path=self.local_model_path,
            local_tokenizer_path=self.local_tokenizer_path,
            model_name=self.model_name,
            openai_creds_key_file=self.openai_creds_key_file,
            local_port=self.local_port,
            local_api_type=self.local_api_type,
            no_chat_template=self.no_chat_template,
            num_samples=self.num_samples,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_logprobs=self.top_logprobs,
            max_request_time=self.max_request_time,
            sleep_time=self.sleep_time,
            echo=self.echo,
            # method-specific parameters
            openai_creds_key_file_2=self.minkprob_openai_creds_key_file_2,
            local_port_2=self.minkprob_local_port_2,
            model_name_2=self.minkprob_model_name_2,
            # local_model_path_2=self.local_model_path_2,  # Currently it is not contained in argument list. Uncomment this when you need it.
            # local_tokenizer_path_2=self.local_tokenizer_path_2,
            do_infer=self.minkprob_do_infer,
        )

    # to use with a vLLM instance
    def contamination_cdd(self):
        main_cdd(
            eval_data=self.eval_data,
            # closed_data parameters
            local_model_path=self.local_model_path,
            local_tokenizer_path=self.local_tokenizer_path,
            model_name=self.model_name,
            openai_creds_key_file=self.openai_creds_key_file,
            local_port=self.local_port,
            local_api_type=self.local_api_type,
            no_chat_template=self.no_chat_template,
            num_samples=self.num_samples,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_logprobs=self.top_logprobs,
            max_request_time=self.max_request_time,
            sleep_time=self.sleep_time,
            echo=self.echo,
            # method-specific parameters
            alpha=self.cdd_alpha,
            xi=self.cdd_xi
        )

    # to use with a vLLM instance
    def contamination_ts_guessing_question_based(self):
        main_ts_guessing_question_based(
            eval_data=self.eval_data,
            eval_data_name=self.eval_data_name,
            n_eval_data_points=self.n_eval_data_points,
            # closed_data parameters
            local_model_path=self.local_model_path,
            local_tokenizer_path=self.local_tokenizer_path,
            model_name=self.model_name,
            openai_creds_key_file=self.openai_creds_key_file,
            local_port=self.local_port,
            local_api_type=self.local_api_type,
            no_chat_template=self.no_chat_template,
            num_samples=self.num_samples,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_logprobs=self.top_logprobs,
            max_request_time=self.max_request_time,
            sleep_time=self.sleep_time,
            echo=self.echo,
            # method-specific parameters
            type_hint=self.ts_guessing_type_hint,
            category_hint=self.ts_guessing_category_hint,
            url_hint=self.ts_guessing_url_hint
        )

    # to use with a vLLM instance
    def contamination_ts_guessing_question_multichoice(self):
        main_ts_guessing_question_multichoice(
            eval_data=self.eval_data,
            eval_data_name=self.eval_data_name,
            n_eval_data_points=self.n_eval_data_points,
            # closed_data parameters
            local_model_path=self.local_model_path,
            local_tokenizer_path=self.local_tokenizer_path,
            model_name=self.model_name,
            openai_creds_key_file=self.openai_creds_key_file,
            local_port=self.local_port,
            local_api_type=self.local_api_type,
            no_chat_template=self.no_chat_template,
            num_samples=self.num_samples,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_logprobs=self.top_logprobs,
            max_request_time=self.max_request_time,
            sleep_time=self.sleep_time,
            echo=self.echo,
        )
