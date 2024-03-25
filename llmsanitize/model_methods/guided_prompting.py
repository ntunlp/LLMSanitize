"""
This file implements the model contamination detection through guided prompting.
https://arxiv.org/pdf/2308.08493.pdf
"""

import nltk
import random
import numpy as np
from rouge_score import rouge_scorer
from datasets import Value
from functools import partial
from llmsanitize.utils.utils import seed_everything, fill_template
from llmsanitize.utils.logger import get_child_logger
from llmsanitize.model_methods.llm import LLM
import llmsanitize.prompts.guided_prompting.general_instructions as gi_prompts
import llmsanitize.prompts.guided_prompting.guided_instructions as gui_prompts

logger = get_child_logger("guided_prompting")


def guided_prompt_filter_fn(example, text_key):
    text = example[text_key]
    # 1. split text to sentences -->
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 2:
        return False
    return True

def guided_prompt_process_fn(
    example,
    idx,
    llm,
    split_name,
    dataset_name,
    label_key,
    text_key,
    general_template,
    guided_template
):
    label = str(example[label_key])
    text = example[text_key]
    seed_everything(idx)
    # 1. split text to sentences -->
    sentences = nltk.sent_tokenize(text)
    # 2. randomly split sentences to two parts
    first_part_length = random.randint(1, len(sentences) - 1)
    first_part = ''.join(sentences[:first_part_length])
    second_part = ''.join(sentences[first_part_length:])
    # query llm
    vars_map = {"split_name": split_name, "dataset_name": dataset_name, "first_piece": first_part, "label": label}
    general_prompt = fill_template(general_template, vars_map)
    guided_prompt = fill_template(guided_template, vars_map)
    general_response, cost = llm.query(general_prompt)
    guided_response, cost_ = llm.query(guided_prompt)
    # get scores
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    general_score = scorer.score(second_part, general_response)['rougeL'].fmeasure
    guided_score = scorer.score(second_part, guided_response)['rougeL'].fmeasure

    # return
    example['general_score'] = general_score
    example['guided_score'] = guided_score
    example['general_response'] = general_response
    example['guided_response'] = guided_response
    example['first_part'] = first_part
    example['second_part'] = second_part
    return example

def main_guided_prompting(
    guided_prompting_task_type,
    eval_data,
    eval_data_name,
    eval_set_key,
    text_key,
    label_key,
    local_port,
    model_name,
    num_proc
):
    ## only examine eval data here
    process_fn = guided_prompt_process_fn

    # based on task type, choose prompt template
    type_str = guided_prompting_task_type
    guided_template = getattr(gui_prompts, f"GUI_{type_str}")
    general_template = getattr(gi_prompts, f"GI_{type_str}")

    # process selected examples parallely
    num_examples_to_test = 800
    random_examples = eval_data.shuffle(seed=42).filter(
        partial(guided_prompt_filter_fn, text_key=text_key)) \
        .filter(lambda _, idx: idx < num_examples_to_test, with_indices=True)

    llm = LLM(local_port=local_port, model_name=model_name)
    process_fn = partial(
        process_fn,
        llm=llm,
        split_name=eval_set_key,
        dataset_name=eval_data_name,
        label_key=label_key,
        text_key=text_key,
        general_template=general_template,
        guided_template=guided_template
    )

    # somehow I need to do this to avoid datasets bug (https://github.com/huggingface/datasets/issues/6020#issuecomment-1632803184)
    features = eval_data.features
    features['general_score'] = Value(dtype='float64', id=None)
    features['guided_score'] = Value(dtype='float64', id=None)
    features["general_response"] = Value(dtype='string', id=None)
    features["guided_response"] = Value(dtype='string', id=None)
    features["first_part"] = Value(dtype='string', id=None)
    features["second_part"] = Value(dtype='string', id=None)

    processed_examples = random_examples.map(process_fn, with_indices=True, num_proc=num_proc,
                                             features=features) \
        .filter(lambda example: len(example['general_response']) > 0 and len(example['guided_response']) > 0)

    scores_diff = [example['guided_score'] - example['general_score'] for example in processed_examples]
    logger.info(f"Tested {len(processed_examples)} examples with guided-prompting for model {model_name}")
    logger.info(f"guided_score - general_score (RougeL)\nmean: {np.mean(scores_diff):.2f}, std: {np.std(scores_diff):.2f}")
    # TODO: add significance measure and bootstrap resampling
    logger.info("skipping the bootstrap resampling and significance measure for now")
