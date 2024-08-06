"""
This file implements the string-matching done for open_data contamination as in PaLM's paper.
https://arxiv.org/pdf/2204.02311.pdf
"""

import numpy as np

from llmsanitize.utils.string_utils import *
from llmsanitize.utils.string_utils_streaming import *
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("palm")


def main_palm(
    train_data,
    eval_data,
    train_data_name,
    eval_data_name,
    eval_set_key,
    stream_train_data=False,
    text_key=None,
    text_keys=None
):
    eval_data = eval_data["text"]

    ngram_size = 8
    if not (stream_train_data):
        train_data = train_data["text"]
        train_ngrams = build_ngrams(train_data, ngram_size)
    else:
        train_ngrams = build_ngrams_streaming(train_data, ngram_size, text_processing_method=None, text_key=text_key, text_keys=text_keys)
    logger.info(f"There are {len(train_ngrams.keys())} {ngram_size}-grams strings in the training set")

    overlap_thresh = 70
    ngram_overlaps = overlap_ngrams(eval_data, train_ngrams, ngram_size, None)

    overlaps = np.array([100 * x[0] / x[1] for x in ngram_overlaps])
    contaminated = np.array([int(x >= overlap_thresh) for x in overlaps])
    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    logger.info(f"Open-data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: ratio of contaminated {ngram_size}-grams is above {overlap_thresh}% (PaLM style open_data contamination)")
    logger.info(f"# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%")
