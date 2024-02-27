from llmsanitize.utils.openai_api_utils import (
    initialize_openai,
    initialize_openai_local,
    query_llm_api,
)
# from llmsanitize.configs.config import *

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LLM:
    def __init__(self, config):
        """
        :param config: config object
            required fields:
                For openai model initialization:
                    - openai.creds_key_file
                For vllm-based model initialization:
                    - local.port
                    - local.model_path
                    - local.tokenizer_path
                Request parameters:
                    - query.model_name  # TODO: Maybe we can simply move these parameters to a parent field so that we can pass all of them at once.
                    - query.num_samples
                    - query.max_tokens
                    - query.top_logprobs
                    - query.max_request_time
                    - query.sleep_time
        """
        # Commented by Fangkai: we should put config as inputs or let the class be able to receive more parameters,
        #   so that we can initialize two models at the same time.
        self.config = config
        if config.openai:
            assert config.openai.creds_key_file, "Please provide the path to your OpenAI API key."
            initialize_openai(config)
            self.query_fn = query_llm_api
            self.api_base = True
        elif config.local.port:
            assert config.local.port, "Please provide the url port to access your local model service."
            initialize_openai_local(config)
            self.query_fn = query_llm_api
            self.config.openai.model_name = self.config.local.model_path
            self.api_base = True
        else:
            assert config.local.model_path and config.local.tokenizer_path, "Please provide the path to your local model and tokenizer."
            self.model = AutoModelForCausalLM.from_pretrained(self.config.local.model_path, torch_dtype=torch.float16).to(self.config.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.local.tokenizer_path)
            self.api_base = False

    def query(self, prompt):
        if self.api_base:
            outputs, cost = self.query_fn(self.config, prompt)
            assert len(outputs) == 1
            return outputs[0], cost
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model.generate(**inputs, max_length=512, num_return_sequences=1)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True), 0

    def batch_query(self, prompts):
        # TODO: Current implementation requires deploy a vllm service first?
        #   Maybe we could also add online inference for better speed.
        outputs, cost = self.query_fn(self.config, prompts)
        return outputs, cost
