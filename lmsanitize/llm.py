from lmsanitize.utils.openai_api_utils import *
# from lmsanitize.configs.config import *

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LLM:
    def __init__(self, config, use_local_model=False):
        # Commented by Fangkai: we should put config as inputs or let the class be able to receive more parameters,
        #   so that we can initialize two models at the same time.
        self.config = config
        if not use_local_model:
            initialize_openai(config)
            self.query_fn = query_llm_api
            self.api_base = True
        else:
            if self.config.api_base:
                initialize_openai_local(config)
                self.query_fn = query_llm_api
                self.config.openai.model_name = self.config.local.model_path
                self.api_base = True
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.config.local.model_path, torch_dtype=torch.float16).to(self.config.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.local.model_path)
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
