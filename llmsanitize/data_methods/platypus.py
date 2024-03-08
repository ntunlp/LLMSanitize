"""
This file implements the embeddings similarity done for data contamination as in Platypus paper.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Following the logic in Platypus paper: https://arxiv.org/pdf/2308.07317.pdf section 2.2
def main_platypus(
    train_data,
    eval_data,
    train_data_name,
    eval_data_name,
    eval_set_key
):
    ## only keep the content per data example, discard labels
    train_data = train_data["text"]
    eval_data = eval_data["text"]

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    train_embeddings = model.encode(train_data)
    eval_embeddings = model.encode(eval_data)
    cos = cosine_similarity(eval_embeddings, train_embeddings)

    thresh = 0.8
    contaminated = (np.max(cos, axis=1) >= thresh).astype(int)
    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    message = f"\nData contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)"
    message += f"\nMethod: Sentence-Transformers embeddings cosine above {thresh} (Platypus style)"
    message += f"\n# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%"
    print(message)