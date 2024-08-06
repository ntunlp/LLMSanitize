"""
This file implements the string-matching done for open_data contamination as in GPT-2's paper.
https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
"""

import re
import numpy as np

from llmsanitize.utils.string_utils import *
from llmsanitize.utils.string_utils_streaming import *
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("gpt2")


def clean_text_gpt2(text):
    text = text.lower()  # lower case
    text = ''.join(i if (i.isalpha() or i == " ") else '' for i in text)  # keep alphanumeric characters
    text = re.sub(' +', ' ', text)  # only single spaces
    text = text.strip()  # initial and final spaces

    return text

def main_gpt2(
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
    if not(stream_train_data):
        train_data = train_data["text"]
        train_ngrams = build_ngrams(train_data, ngram_size, clean_text_gpt2)
    else:
        train_ngrams = build_ngrams_streaming(train_data, ngram_size, clean_text_gpt2, text_key, text_keys)
    logger.info(f"There are {len(train_ngrams.keys())} {ngram_size}-grams in the training set")

    ngram_overlaps = overlap_ngrams(eval_data, train_ngrams, ngram_size, clean_text_gpt2)

    contaminated = np.array([int(x[0] > 0) for x in ngram_overlaps])
    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    overlaps = np.array([100 * x[0]/x[1] for x in ngram_overlaps])
    mean_overlap = np.mean(overlaps)
    logger.info(f"Open-data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: matching of {ngram_size}-grams (GPT-2 style open_data contamination)")
    logger.info(f"# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%")
    logger.info(f"Mean {ngram_size}-grams overlap: {mean_overlap:.4f}%")
