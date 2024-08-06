"""
This file implements the embeddings similarity done for open_data contamination as in Platypus paper.
https://arxiv.org/pdf/2308.07317.pdf
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llmsanitize.utils.string_utils_streaming import *
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("platypus")


def main_platypus(
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

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    eval_embeddings = model.encode(eval_data)
    if not (stream_train_data):
        train_data = train_data["text"]
        train_embeddings = model.encode(train_data)
    else:
        train_embeddings = build_embeddings_streaming(
            train_data,
            model,
            bufer_size=10000,
            text_processing_method=None,
            text_key=text_key,
            text_keys=text_keys
        )

    cos = cosine_similarity(eval_embeddings, train_embeddings)

    thresh = 0.8
    contaminated = (np.max(cos, axis=1) >= thresh).astype(int)
    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    logger.info(f"Open-data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: Sentence-Transformers embeddings cosine above {thresh} (Platypus style)")
    logger.info(f"# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%")
