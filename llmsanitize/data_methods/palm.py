"""
This file implements the string-matching done for data contamination as in PaLM's paper.
"""

import numpy as np
from llmsanitize.utils.string_utils import *


# Following the logic in PaLM's paper: https://arxiv.org/pdf/2204.02311.pdf section 8
def main_palm(
    train_data,
    eval_data,
    train_data_name,
    eval_data_name,
    eval_set_key
):
    ## only keep the content per data example, discard labels
    train_data = train_data["text"]
    eval_data = eval_data["text"]

    ngram_size = 8
    train_ngrams = build_ngrams(train_data, ngram_size, None)
    message = f"There are {len(train_ngrams.keys())} {ngram_size}-grams strings in the training set"
    print(message)

    overlap_thresh = 70
    ngram_overlaps = overlap_ngrams(eval_data, train_ngrams, ngram_size, None)
    overlaps = np.array([100 * x[0] / x[1] for x in ngram_overlaps])
    contaminated = np.array([int(x >= overlap_thresh) for x in overlaps])
    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    message = f"\nData contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)"
    message += f"\nMethod: ratio of contaminated {ngram_size}-grams is above {overlap_thresh}% (PaLM style data contamination)"
    message += f"\n# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%"
    print(message)