"""
This file implements the string-matching based on METEOR recall described in this paper:
https://arxiv.org/pdf/2310.17589.pdf
"""

import evaluate

from llmsanitize.utils.string_utils import *
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("meteor")


def main_meteor(
    train_data,
    eval_data,
    train_data_name,
    eval_data_name,
    eval_set_key
):
    train_data = train_data["text"]
    eval_data = eval_data["text"]

    meteor = evaluate.load('meteor')
    gamma = 0.8
    thresh = 0.75

    contaminated = []
    for i in tqdm(range(len(eval_data))):
        tagged = 0
        length = len(eval_data[i])
        train_points = [x for x in train_data if len(x) >= (2*length)]
        for j in range(len(train_points)):
            sequences = []
            for k in range(len(train_points[j])-(2*length)):
                sequence = train_points[j][k:(k + 2*length)]
                sequences.append(sequence)
            results = meteor.compute(predictions=[eval_data[i]], references=sequences, gamma=gamma)
            if results['meteor'] >= thresh:
                tagged = 1
                break
        contaminated.append(tagged)

    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    logger.info(f"Data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: exact string matching")
    logger.info(f"# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%")