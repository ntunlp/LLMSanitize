import copy

from llmsanitize.utils.openai_api_utils import (
    initialize_openai,
    initialize_openai_local,
    query_llm_api,
)
# from llmsanitize.configs.config import *

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from llmsanitize.utils.utils import dict_to_object
from argparse import Namespace
from llmsanitize.configs.config import config


class LLM:
    def __init__(self,
                 openai_creds_key_file: str = None,
                 local_port: str = None,
                 local_model_path: str = None,
                 local_tokenizer_path: str = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 model_name: str = None,
                 num_samples: int = 1,
                 max_tokens: int = 128,
                 top_logprobs: int = 0,
                 max_request_time: int = 600,
                 sleep_time: int = 1):
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
        # self.config = config
        if local_model_path:
            print(f"Loading local model from {local_model_path} and tokenizer from {local_tokenizer_path}.")
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype="auto").to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
            self.api_base = False
        elif local_port:
            print(f"Initializing vllm service from port {local_port}.")
            _config = dict_to_object({"local": {"port": local_port}})
            initialize_openai_local(_config)
            self.query_fn = query_llm_api
            self.api_base = True
        else:
            print("Initializing OpenAI API.")
            _config = dict_to_object({"openai": {"creds_key_file": openai_creds_key_file}})
            initialize_openai(_config)
            self.query_fn = query_llm_api
            self.api_base = True
        # if config.openai:
        #     assert config.openai.creds_key_file, "Please provide the path to your OpenAI API key."
        #     initialize_openai(config)
        #     self.query_fn = query_llm_api
        #     self.api_base = True
        # elif config.local.port:
        #     assert config.local.port, "Please provide the url port to access your local model service."
        #     initialize_openai_local(config)
        #     self.query_fn = query_llm_api
        #     self.config.openai.model_name = self.config.local.model_path
        #     self.api_base = True
        # else:
        #     assert config.local.model_path and config.local.tokenizer_path, "Please provide the path to your local model and tokenizer."
        #     self.model = AutoModelForCausalLM.from_pretrained(self.config.local.model_path, torch_dtype=torch.float16).to(self.config.device)
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.config.local.tokenizer_path)
        #     self.api_base = False

        _query_config = {
            "local": {
                "port": local_port,
            },
            "openai": {
                "creds_key_file": openai_creds_key_file,
                "model_name": model_name,
            },
            "query": {
                "num_samples": num_samples,
                "max_tokens": max_tokens,
                "top_logprobs": top_logprobs,
                "max_request_time": max_request_time,
                "sleep_time": sleep_time,
            }
        }
        self._query_config = dict_to_object(_query_config)
        print("====================== Query Config =======================")
        print(_query_config)

    def query(self, prompt, return_full_response: bool = False):
        if self.api_base:
            outputs, full_response, cost = self.query_fn(self._query_config, prompt)
            assert len(outputs) == 1
            if return_full_response:
                return outputs[0], full_response, cost
            return outputs[0], cost
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model.generate(**inputs, max_length=512, num_return_sequences=1)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True), 0

    def batch_query(self, prompts):
        # TODO: Current implementation requires deploy a vllm service first?
        #   Maybe we could also add online inference for better speed.
        outputs, cost = self.query_fn(self._query_config, prompts)
        return outputs, cost

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(
            openai_creds_key_file=getattr(args, "openai_creds_key_file", config.openai.creds_key_file),
            local_port=args.local_port,
            local_model_path=args.local_model_path,
            local_tokenizer_path=getattr(args, "local_tokenizer_path", args.local_model_path),
            model_name=args.model_name,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            top_logprobs=args.top_logprobs,
            max_request_time=args.max_request_time,
            sleep_time=args.sleep_time
        )
