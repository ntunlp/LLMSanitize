"""
This file includes method-specific utilization functions
"""

import random
from lmsanitize.utils.utils import seed_everything, fill_template
from lmsanitize.llm import LLM
import nltk
from copy import copy

def guided_prompt_process_fn(example, idx, config, use_local_model, split_name, 
                                dataset_name, label_key, text_key, general_template, guided_template):
    label = str(example[label_key])
    text = example[text_key]
    seed_everything(idx)
    # 1. split text to sentences --> 2. randomly split sentences to two parts
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 2:
        return None
    first_part_length = random.randint(1, len(sentences)-1)
    first_part = ''.join(sentences[:first_part_length])
    second_part = ''.join(sentences[first_part_length:])
    # query llm
    vars_map = {"split_name":split_name, "dataset_name":dataset_name, "first_piece":first_part, "label": label}
    general_prompt = fill_template(general_template, vars_map)
    guided_prompt = fill_template(guided_template, vars_map)
    llm = LLM(use_local_model)
    general_response, cost = llm.query(general_prompt)
    guided_response, cost_ = llm.query(guided_prompt)
    import pdb;pdb.set_trace()