"""
This file implements the closed_data contamination detection through guided prompting.
https://arxiv.org/pdf/2308.08493.pdf
"""

import nltk
import random
import numpy as np
from rouge_score import rouge_scorer
from datasets import Value
from functools import partial
from llmsanitize.utils.utils import seed_everything, fill_template
from llmsanitize.utils.logger import get_child_logger, suspend_logging
from llmsanitize.closed_data_methods.llm import LLM
import llmsanitize.prompts.guided_prompting.general_instructions as gi_prompts
import llmsanitize.prompts.guided_prompting.guided_instructions as gui_prompts
from scipy.stats import bootstrap

logger = get_child_logger("guided_prompting")


def guided_prompt_split_fn(example, idx, dataset_name, text_key):
    ''' split content per example to part 1 and part 2
        For AGnews: split ['text'] into 2 parts
            ARC: split ['question']+['choices']
    '''
    seed_everything(idx)
    splits = {'guided_prompt_part_1': '', 'guided_prompt_part_2': ''}
    # split the input field to two parts
    if dataset_name in ['ag_news', 'gsm8k', 'cais/mmlu']:
        text = example[text_key]
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return splits
        first_part_length = random.randint(1, len(sentences) - 1)
        splits['guided_prompt_part_1'] = ''.join(sentences[:first_part_length])
        splits['guided_prompt_part_2'] = ''.join(sentences[first_part_length:])
    
    # split to question + choices[0], choices[1:]
    elif dataset_name in ['allenai/ai2_arc']:
        choices = example['choices']
        choices = [_label+'.'+_text for _text, _label in zip(choices['text'], choices['label'])]
        splits['guided_prompt_part_1'] = example[text_key] + '\n' + choices[0]
        splits['guided_prompt_part_2'] = '\n'.join(choices[1:])
    
    # NLI tasks
    elif dataset_name in ['Rowan/hellaswag']:
        splits['guided_prompt_part_1'] = example[text_key]
        splits['guided_prompt_part_2'] = example['endings'][int(example['label'])] 
    elif dataset_name in ['truthful_qa']:
        splits['guided_prompt_part_1'] = example[text_key]
        splits['guided_prompt_part_2'] = example['best_answer']
    elif dataset_name == "winogrande":
        sents = example[text_key].split('_')
        splits['guided_prompt_part_1'] = sents[0]
        splits['guided_prompt_part_2'] = sents[1]
    else:
        raise(f"Error! guided_prompt_split_fn not found processing for dataset_name: {dataset_name}")

    return splits

def guided_prompt_process_label(example, dataset_name):
    if dataset_name == 'cais/mmlu':
        example['answer_text'] = example['choices'][int(example['answer'])]
    elif dataset_name == 'winogrande':
        example['answer_token'] = example['option1'] + '/' + example['option2']
    
    return example

def bootstrap_test(data):
    ''' bootstrap test (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
        to check if to reject the H0 hypothesis that there's no difference between guided-prompt and general-prompt
    Args:
        data: a sequence of score difference (s_guided - s_general)
    Return:
        p-value of diff <= 0
    '''
    res = bootstrap((data,), np.mean, n_resamples=10000)

    return (res.bootstrap_distribution <= 0.).sum() / 10000.

@suspend_logging
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
    seed_everything(idx)
    label = str(example[label_key])
    first_part = example['guided_prompt_part_1']
    second_part = example['guided_prompt_part_2']

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
    eval_data,
    eval_data_name,
    eval_set_key,
    text_key,
    label_key,
    num_proc,
    # closed_data parameters
    local_model_path: str = None,
    local_tokenizer_path: str = None,
    model_name: str = None,
    openai_creds_key_file: str = None,
    local_port: str = None,
    local_api_type: str = None,
    no_chat_template: bool = False,
    num_samples: int = 1,
    max_input_tokens: int = 512,
    max_output_tokens: int = 128,
    temperature: float = 0.0,
    top_logprobs: int = 0,
    max_request_time: int = 600,
    sleep_time: int = 1,
    echo: bool = False,
    # method-specific parameters
    guided_prompting_task_type: str = None,
):
    # based on task type, choose prompt template
    type_str = guided_prompting_task_type
    guided_template = getattr(gui_prompts, f"GUI_{type_str}")
    general_template = getattr(gi_prompts, f"GI_{type_str}")

    # process selected examples parallely
    num_examples_to_test = 100
    split_fn = partial(guided_prompt_split_fn, dataset_name=eval_data_name, text_key=text_key)
    label_fn = partial(guided_prompt_process_label, dataset_name=eval_data_name)
    print(eval_data)
    eval_data = eval_data.map(split_fn, num_proc=num_proc, load_from_cache_file=False, with_indices=True)\
        .filter(lambda example: len(example['guided_prompt_part_1']) > 0 and len(example['guided_prompt_part_2']) > 0)\
        .map(label_fn, num_proc=num_proc)
    random_examples = eval_data.shuffle(seed=42).filter(lambda _, idx: idx < num_examples_to_test, with_indices=True)
    print(f"random samples: {random_examples}")
    
    llm = LLM(
        local_model_path=local_model_path,
        local_tokenizer_path=local_tokenizer_path,
        model_name=model_name,
        openai_creds_key_file=openai_creds_key_file,
        local_port=local_port,
        local_api_type=local_api_type,
        no_chat_template=no_chat_template,
        num_samples=num_samples,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_logprobs=top_logprobs,
        max_request_time=max_request_time,
        sleep_time=sleep_time,
        echo=echo,
    )

    process_fn = partial(
        guided_prompt_process_fn,
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

    processed_examples = random_examples.map(
        process_fn, with_indices=True, num_proc=num_proc, features=features, load_from_cache_file=False
    )
    processed_examples = [example for example in processed_examples if (len(example['general_response']) > 0) and (len(example['guided_response']) > 0)]

    scores_diff = [example['guided_score'] - example['general_score'] for example in processed_examples]
    logger.info(f"Tested {len(processed_examples)} examples with guided-prompting for closed_data {model_name}")
    # TODO: add significance measure and bootstrap resampling
    p_value = bootstrap_test(scores_diff)
    logger.info(f"dataset: {eval_data_name}, guided_score - general_score (RougeL)")
    logger.info(f"mean: {np.mean(scores_diff):.3f}, std: {np.std(scores_diff):.3f}, p-value of diff <= 0: {p_value:.3f}")
