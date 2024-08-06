"""
This file implements the exact match method for open_data contamination detection, as done in this paper:
https://arxiv.org/pdf/2104.08758.pdf
"""

import re
import numpy as np

from llmsanitize.utils.string_utils import *
from llmsanitize.utils.string_utils_streaming import *
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
    eval_set_key,
    stream_train_data=False,
    text_key=None,
    text_keys=None
):
    eval_data = eval_data["text"]

    if not (stream_train_data):
        train_data = train_data["text"]
        train_strings = build_full_strings(train_data, clean_text_exact)
    else:
        train_strings = build_full_strings_streaming(train_data, clean_text_exact, text_key, text_keys)

    contaminated = []
    for i in tqdm(range(len(eval_data))):
        eval_data_point = clean_text_exact(eval_data[i])
        tagged = 0
        for k in train_strings.keys():
            if eval_data_point in k:
                tagged = 1
                break
        contaminated.append(tagged)
    contaminated = np.array(contaminated)

    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    logger.info(f"Open-data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: exact string matching")
    logger.info(f"# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%")
