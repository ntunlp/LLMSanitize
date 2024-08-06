"""
This file implements Contamination Detection via output Distribution for LLMs (CDD)
https://arxiv.org/pdf/2402.15938.pdf
"""

import numpy as np
from tqdm import tqdm

from llmsanitize.closed_data_methods.llm import LLM
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("cdd")


def get_ed(a, b):
    if len(b) == 0:
        return len(a)
    elif len(a) == 0:
        return len(b)
    else:
        dist = np.zeros((len(a)+1, len(b)+1))
        for i in range(len(a)):
            for j in range(len(b)):
                if a[i] == b[j]:
                    dist[i+1, j+1] = dist[i, j]
                else:
                    dist[i+1, j+1] = 1 + min(dist[i, j], dist[i+1, j], dist[i, j+1])
        
        return int(dist[-1, -1])

def get_peak(samples, s_0, alpha):
    lengths = [len(x) for x in samples]
    l = min(lengths)
    l = min(l, 100)
    thresh = int(np.ceil(alpha * l))
    distances = [get_ed(s, s_0) for s in samples]
    rhos = [len([x for x in distances if x == d]) for d in range(0, thresh+1)]
    peak = sum(rhos)

    return peak

def inference(
    eval_data,
    llm0,
    llm,
    num_samples,
    alpha,
    xi
):
    cdd_results = []
    for data_point in tqdm(eval_data):
        prompt = data_point["text"]

        _, response_0, _ = llm0.query(prompt, return_full_response=True)
        s_0 = response_0["choices"][0]["text"]

        _, responses, _ = llm.query(prompt, return_full_response=True)
        samples = [responses["choices"][j]["text"] for j in range(num_samples)]

        peak = get_peak(samples, s_0, alpha)
        leaked = int(peak > xi)
        cdd_results.append(leaked)
    cdd_results = np.array(cdd_results)

    return cdd_results

def main_cdd(
    eval_data,
    # closed_data parameters
    local_model_path: str = None,
    local_tokenizer_path: str = None,
    model_name: str = None,
    openai_creds_key_file: str = None,
    local_port: str = None,
    local_api_type: str = None,
    no_chat_template: bool = False,
    num_samples: int = 20,
    max_input_tokens: int = 512,
    max_output_tokens: int = 128,
    temperature: float = 0.8,
    top_logprobs: int = 0,
    max_request_time: int = 600,
    sleep_time: int = 1,
    echo: bool = False,
    # method-specific parameters
    alpha: float = 0.05,
    xi: float = 0.01,
):
    llm0 = LLM(
        local_model_path=local_model_path,
        local_tokenizer_path=local_tokenizer_path,
        model_name=model_name,
        openai_creds_key_file=openai_creds_key_file,
        local_port=local_port,
        local_api_type=local_api_type,
        no_chat_template=no_chat_template,
        num_samples=1,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        top_logprobs=top_logprobs,
        max_request_time=max_request_time,
        sleep_time=sleep_time,
        echo=echo
    )

    llm = LLM(
        local_model_path=local_model_path,
        local_tokenizer_path=local_tokenizer_path,
        model_name=model_name,
        openai_creds_key_file=openai_creds_key_file,
        local_port=local_port,
        local_api_type=local_api_type,
        no_chat_template=no_chat_template,
        num_samples=num_samples,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_logprobs=top_logprobs,
        max_request_time=max_request_time,
        sleep_time=sleep_time,
        echo=echo
    )

    logger.info(f"all data size: {len(eval_data)}")

    cdd_results = inference(eval_data, llm0, llm, num_samples, alpha, xi)

    contaminated_frac = 100 * np.mean(cdd_results)
    logger.info(f"Checking contamination level of model {local_model_path} with CDD")
    logger.info(f"Contamination level: {contaminated_frac:.4f}% of data points")
