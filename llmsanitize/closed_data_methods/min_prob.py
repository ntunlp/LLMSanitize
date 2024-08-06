"""
This file implements the closed_data contamination detection through the min-K-prob approach.
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

from llmsanitize.closed_data_methods.llm import LLM
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("min_prob")

LLM1: LLM
LLM2: LLM

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# plot open_data
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    score = np.nan_to_num(score)
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)

    return fpr, tpr, auc(fpr, tpr), acc

def do_plot(
    prediction,
    answers,
    sweep_fn=sweep,
    metric='auc',
    legend="",
    output_dir=None
):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr < .05)[0][-1]]
    # bp()
    logger.info('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n' % (legend, auc, acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f' % auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f' % acc

    plt.plot(fpr, tpr, label=legend + metric_text)

    return legend, auc, acc, low

def fig_fpr_tpr(all_output, output_dir, do_infer: bool = False):
    logger.info(f"Min-Prob method output_dir: {output_dir}")
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        if "label" in ex and ex["label"] and not do_infer:
            answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])

    plt.figure(figsize=(4, 3))
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            if answers:
                legend, auc, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
                f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n' % (legend, auc, acc, low))

            scores = np.nan_to_num(predictions)
            logger.info(f"=================== {metric} {np.mean(scores):.4f}")
            f.write(f"{metric} {np.mean(scores):.4f}\n")

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")

def calculate_perplexity(prompt, llm: LLM):
    prompt = prompt.replace('\x00', '')
    _, responses, _ = llm.query(prompt, return_full_response=True)
    data = responses["choices"][0]["logprobs"]
    all_prob = [d for d in data["token_logprobs"][:-llm.query_config.query.max_tokens] if d is not None]
    p1 = np.exp(-np.mean(all_prob))

    return p1, all_prob, np.mean(all_prob)

def inference(llm1: LLM, llm2: LLM, _input):
    text = _input["text"]
    pred = {}

    p1, all_prob, p1_likelihood = calculate_perplexity(text, llm1)
    p_lower, _, p_lower_likelihood = calculate_perplexity(text.lower(), llm1)

    p_ref, all_prob_ref, p_ref_likelihood = calculate_perplexity(text, llm2)

    # ppl
    pred["ppl"] = p1
    # Ratio of log ppl of large and small models
    pred["ppl/Ref_ppl (calibrate PPL to the reference closed_data)"] = p1_likelihood - p_ref_likelihood

    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1) / zlib_entropy
    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio * 100}% Prob"] = -np.mean(topk_prob).item()

    _input["pred"] = pred

    return _input

def _client_init(llm1, llm2):
    global LLM1, LLM2
    LLM1 = llm1
    LLM2 = llm2

def _process_fn(x):
    return inference(LLM1, LLM2, x)

def main_min_prob(
    eval_data,
    num_proc: int = 8,
    output_dir: str = "output",
    # closed_data parameters
    local_model_path: str = None,
    local_tokenizer_path: str = None,
    model_name: str = None,
    openai_creds_key_file: str = None,
    local_port: str = None,
    local_api_type: str = None,
    no_chat_template: bool = False,
    num_samples: int = 1,
    max_input_tokens: int = 512,
    max_output_tokens: int = 128,
    temperature: float = 0.0,
    top_logprobs: int = 0,
    max_request_time: int = 600,
    sleep_time: int = 1,
    echo: bool = False,
    # method-specific parameters
    openai_creds_key_file_2: str = None,
    local_port_2: str = None,
    model_name_2: str = None,
    local_model_path_2: str = None,
    local_tokenizer_path_2: str = None,
    do_infer: bool = False,
):
    
    llm1 = LLM(
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
        echo=echo,
    )

    llm2 = LLM(
        local_model_path=local_model_path_2,
        local_tokenizer_path=local_tokenizer_path_2,
        model_name=model_name_2,
        openai_creds_key_file=openai_creds_key_file_2,
        local_port=local_port_2,
        local_api_type=local_api_type,
        no_chat_template=no_chat_template,
        num_samples=num_samples,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_logprobs=top_logprobs,
        max_request_time=max_request_time,
        sleep_time=sleep_time,
        echo=echo,
    )

    logger.info(f"all data size: {len(eval_data)}")

    if num_proc <= 1:
        all_output = []
        for text in tqdm(eval_data):
            # logger.info(text)
            new_ex = inference(llm1, llm2, text)  # Here, `test_data` is Dataset, and `text` is a dictionary.
            all_output.append(new_ex)
    else:
        with Pool(num_proc, initializer=_client_init, initargs=(llm1, llm2,)) as p:
            all_output = list(tqdm(
                p.imap(_process_fn, eval_data, chunksize=32),
                total=len(eval_data),
                desc="Sending requests to local service"
            ))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig_fpr_tpr(all_output, output_dir, do_infer)
