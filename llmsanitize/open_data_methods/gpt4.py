"""
This file implements the string-matching done for open_data contamination as in GPT-4's paper.
https://arxiv.org/pdf/2303.08774.pdf
"""

from llmsanitize.utils.string_utils import *
from llmsanitize.utils.string_utils_streaming import *
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("gpt4")


def clean_text_gpt4(text):
    text = ''.join(i if i.isalpha() else '' for i in text)  # keep alphanumeric characters

    return text

def main_gpt4(
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

    string_size = 50
    if not (stream_train_data):
        train_data = train_data["text"]
        train_substrings = build_substrings(train_data, string_size, clean_text_gpt4)
    else:
        train_substrings = build_substrings_streaming(train_data, string_size, clean_text_gpt4, text_key, text_keys)
    logger.info(f"There are {len(train_substrings.keys())} {string_size}-chars strings in the training set")

    n_samples = 3
    contaminated = overlap_substrings_sample(eval_data, train_substrings, string_size, n_samples, clean_text_gpt4)

    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    logger.info(f"Open-data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: sampling {n_samples} {string_size}-chars substring (GPT-4 style open_data contamination)")
    logger.info(f"# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%")
