"""
This file implements the model contamination detection through the min-K-prob approach.
"""
# Most codes are copied from https://github.com/swj0419/detect-pretrain-code/blob/main/src/run.py

import copy
import os.path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zlib
from collections import defaultdict
from multiprocessing import Pool
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from llmsanitize.model_methods.llm import LLM
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("min_prob")

LLM1: LLM
LLM2: LLM

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# plot data
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    score = np.nan_to_num(score)
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    
    return fpr, tpr, auc(fpr, tpr), acc

def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None):
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

def fig_fpr_tpr(all_output, output_dir):
    logger.info(f"Min-Prob method output_dir: {output_dir}")
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])

    plt.figure(figsize=(4, 3))
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            legend, auc, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
            f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n' % (legend, auc, acc, low))

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
    # print("Response", responses)
    data = responses["choices"][0]["logprobs"]
    all_prob = [d for d in data["token_logprobs"] if d is not None]
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
    pred["ppl/Ref_ppl (calibrate PPL to the reference model)"] = p1_likelihood - p_ref_likelihood

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

# Following the logic from this paper: https://arxiv.org/pdf/2310.16789.pdf
def main_min_prob(
    args,
    test_data
):
    num_proc = args.num_proc
    llm1 = LLM.from_args(args=args)

    tmp_args = copy.deepcopy(args)
    tmp_args.model_name = args.model_name_2
    tmp_args.openai_creds_key_file = args.openai_creds_key_file_2
    tmp_args.local_port = args.local_port_2
    llm2 = LLM.from_args(args=tmp_args)

    logger.info(f"all data size: {len(test_data)}")

    if num_proc <= 1:
        all_output = []
        for text in tqdm(test_data):
            # logger.info(text)
            new_ex = inference(llm1, llm2, text)  # Here, `test_data` is Dataset, and `text` is a dictionary.
            all_output.append(new_ex)
    else:
        with Pool(num_proc, initializer=_client_init, initargs=(llm1, llm2,)) as p:
            all_output = list(tqdm(
                p.imap(_process_fn, test_data, chunksize=32),
                total=len(test_data),
                desc="Sending requests to local service"
            ))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    fig_fpr_tpr(all_output, args.output_dir)