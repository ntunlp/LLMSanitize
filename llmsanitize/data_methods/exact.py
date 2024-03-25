"""
This file implements the exact match method for data contamination detection, as done in this paper:
https://arxiv.org/pdf/2104.08758.pdf
"""

import re
import numpy as np

from llmsanitize.utils.string_utils import *
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("exact")


def clean_text_exact(text):
    text = text.lower()  # lower case
    text = ''.join(i if (i.isalpha() or i == " ") else '' for i in text)  # keep alphanumeric characters
    text = re.sub(' +', ' ', text)  # only single spaces
    text = text.strip()  # initial and final spaces

    return text

def main_exact(
    train_data,
    eval_data,
    train_data_name,
    eval_data_name,
    eval_set_key
):
    train_data = train_data["text"]
    eval_data = eval_data["text"]

    train_items = {}
    for x in train_data:
        train_items[clean_text_exact(x)] = 0

    contaminated = []
    for i in tqdm(range(len(eval_data))):
        eval_data_point = clean_text_exact(eval_data[i])
        tagged = 0
        for k in train_items.keys():
            if eval_data_point in k:
                tagged = 1
                break
        contaminated.append(tagged)
    contaminated = np.array(contaminated)

    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    logger.info(f"Data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: exact string matching")
    logger.info(f"# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%")
