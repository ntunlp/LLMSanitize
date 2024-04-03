"""
This file implements the model contamination detection through the min-K-prob approach.
https://arxiv.org/pdf/2310.16789.pdf
"""
# Most codes are copied from https://github.com/swj0419/detect-pretrain-code/blob/main/src/run.py

import os.path
from collections import defaultdict
from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zlib
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from llmsanitize.model_methods.llm import LLM
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("cdd")


def main_cdd(
    eval_data,
    local_port: str = None,
    local_model_path: str = None,
    local_tokenizer_path: str = None,
    model_name: str = None,
    num_samples: int = 1,
    max_tokens: int = 128,
    top_logprobs: int = 0,
    max_request_time: int = 600,
    sleep_time: int = 1,
    echo: bool = False,
    num_proc: int = 8,
    output_dir: str = "output",
    do_infer: bool = False,
):
    llm0 = LLM(
        local_port=local_port,
        local_model_path=local_model_path,
        local_tokenizer_path=local_tokenizer_path,
        model_name=model_name,
        num_samples=1,
        temperature=0.0
    )

    llm = LLM(
        local_port=local_port,
        local_model_path=local_model_path,
        local_tokenizer_path=local_tokenizer_path,
        model_name=model_name,
        num_samples=num_samples,
        temperature=0.0
    )

    logger.info(f"all data size: {len(eval_data)}")

    for text in tqdm(eval_data):
        prompt = text
        prompt = prompt.replace('\x00', '')
        _, responses, _ = llm.query(prompt, return_full_response=True)
        print(responses)

