"""
This file implements the string-matching done for open_data contamination as in GPT-3's paper.
https://arxiv.org/pdf/2005.14165.pdf
"""

import string
import numpy as np

from llmsanitize.utils.string_utils import *
from llmsanitize.utils.string_utils_streaming import *
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("gpt3")


def clean_text_gpt3(text):
    text = text.lower()  # lower case
    text = ' '.join(word.strip(string.punctuation) for word in text.split())

    return text

def main_gpt3(
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

    ngram_size = 13
    if not (stream_train_data):
        train_data = train_data["text"]
        train_ngrams = build_ngrams(train_data, ngram_size, clean_text_gpt3)
    else:
        train_ngrams = build_ngrams_streaming(train_data, ngram_size, clean_text_gpt3, text_key, text_keys)
    logger.info(f"There are {len(train_ngrams.keys())} {ngram_size}-grams in the training set")

    max_count = 10
    n_removed = 0
    for k in train_ngrams.keys():
        if train_ngrams[k] >= max_count:
            del train_ngrams[k]
            n_removed += 1
    logger.info(f"Removed {n_removed} {ngram_size}-grams being too frequent in the training set")

    n_collisions = 1
    ngram_overlaps = overlap_ngrams(eval_data, train_ngrams, ngram_size, clean_text_gpt3)

    contaminated = np.array([int(x[0] >= n_collisions) for x in ngram_overlaps])
    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    overlaps = np.array([100 * x[0] / x[1] for x in ngram_overlaps])
    mean_overlap = np.mean(overlaps)
    logger.info(f"Open-data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: matching of {ngram_size}-grams (GPT-3 style open_data contamination)")
    logger.info(f"# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%")
    logger.info(f"Mean {ngram_size}-grams overlap: {mean_overlap:.4f}%")
