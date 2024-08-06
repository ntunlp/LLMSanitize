"""
This file implements an LLM class used for closed_data-based contamination detection methods.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmsanitize.utils.post_utils import (
    initialize_post,
    query_llm_post,
)
from llmsanitize.utils.openai_utils import (
    initialize_openai,
    query_llm_openai,
)
from llmsanitize.utils.utils import dict_to_object
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("LLM")


class LLM:
    def __init__(
        self,
        local_model_path: str = None,
        local_tokenizer_path: str = None,
        model_name: str = None,
        openai_creds_key_file: str = None,
        local_port: str = None,
        local_api_type: str = "post",
        no_chat_template: bool = False,
        num_samples: int = 1,
        max_input_tokens: int = 512,
        max_output_tokens: int = 128,
        temperature: float = 0.0,
        top_logprobs: int = 0,
        max_request_time: int = 600,
        sleep_time: int = 1,
        echo: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        :param config: config object
            required fields:
                For local closed_data inference (no API):
                    - local_model_path
                    - local_tokenizer_path
                For vllm-based closed_data initialization:
                    - local_port
                    - model_name
                For openai closed_data initialization:
                    - openai.creds_key_file
                Request parameters:
                    - query.num_samples
                    - query.max_tokens
                    - query.top_logprobs
                    - query.max_request_time
                    - query.sleep_time
        """
        # run local closed_data with no API
        if local_model_path:
            logger.info(f"Loading local closed_data from {local_model_path} and tokenizer from {local_tokenizer_path}.")
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype="auto").to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
            self.api_base = False
        # run local closed_data with the vLLM API
        elif local_port:
            logger.info(f"Initializing vllm service from port {local_port}.")
            _config = dict_to_object({"local": {"port": local_port}})
            initialize_post(_config)
            self.query_fn = query_llm_post
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.api_base = True
        # run closed_data with the OpenAI API, either an OpenAI closed_data, or a local closed_data through vLLM
        else:
            logger.info("Initializing OpenAI API.")
            _config = dict_to_object({"openai": {"creds_key_file": openai_creds_key_file}})
            initialize_openai(_config)
            self.query_fn = query_llm_openai
            self.api_base = True
            if model_name is None:
                model_name = "gpt-3.5-turbo-0125" # fall back to GPT-3.5 as default

        _query_config = {
            "local": {
                "tokenizer": self.tokenizer,
                "port": local_port,
                "api_type": local_api_type,
            },
            "openai": {
                "creds_key_file": openai_creds_key_file,
                "model_name": model_name,
            },
            "query": {
                "no_chat_template": no_chat_template,
                "num_samples": num_samples,
                "max_tokens": max_output_tokens,
                "top_logprobs": top_logprobs,
                "max_request_time": max_request_time,
                "sleep_time": sleep_time,
                "echo": echo,
                "temperature": temperature
            }
        }
        self.query_config = dict_to_object(_query_config)
        logger.info("====================== Query Config =======================")
        logger.info(_query_config)

        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def query(self, prompt, return_full_response: bool = False):
        if self.api_base:
            outputs, full_response, cost = self.query_fn(self.query_config, prompt)
            if return_full_response:
                return outputs[0], full_response, cost
            
            return outputs[0], cost
        else:
            if not(self.query_config.query.no_chat_template):
                prompt = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens
            )
            outputs = self.model.generate(
                **inputs,
                num_return_sequences=1,
                max_new_tokens=self.max_output_tokens,
                temperature=self.temperature
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True), 0

    def batch_query(self, prompts):
        # TODO: Current implementation requires deploy a vllm service first?
        #   Maybe we could also add online inference for better speed.
        outputs, cost = self.query_fn(self.query_config, prompts)
        
        return outputs, cost
