"""
This file implements an LLM class used for model-based contamination detection methods.
"""

import copy
import torch
from argparse import Namespace
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmsanitize.utils.openai_api_utils import (
    initialize_openai,
    initialize_openai_local,
    query_llm_api,
)
from llmsanitize.utils.utils import dict_to_object
from llmsanitize.configs.config import config
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("LLM")


class LLM:
    def __init__(
            self,
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
            sleep_time: int = 1,
            echo: bool = False,
    ):
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
                    - query.model_name
                    - query.num_samples
                    - query.max_tokens
                    - query.top_logprobs
                    - query.max_request_time
                    - query.sleep_time
        """
        if local_model_path:
            logger.info(f"Loading local model from {local_model_path} and tokenizer from {local_tokenizer_path}.")
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype="auto").to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
            self.api_base = False
        elif local_port:
            logger.info(f"Initializing vllm service from port {local_port}.")
            _config = dict_to_object({"local": {"port": local_port}})
            initialize_openai_local(_config)
            self.query_fn = query_llm_api
            self.api_base = True
        else:
            logger.info("Initializing OpenAI API.")
            _config = dict_to_object({"openai": {"creds_key_file": openai_creds_key_file}})
            initialize_openai(_config)
            self.query_fn = query_llm_api
            self.api_base = True

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
                "echo": echo,
            }
        }
        self.query_config = dict_to_object(_query_config)
        logger.info("====================== Query Config =======================")
        logger.info(_query_config)

    def query(self, prompt, return_full_response: bool = False):
        if self.api_base:
            outputs, full_response, cost = self.query_fn(self.query_config, prompt)
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
        outputs, cost = self.query_fn(self.query_config, prompts)
        return outputs, cost
