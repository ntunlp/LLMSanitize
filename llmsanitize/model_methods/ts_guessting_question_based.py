"""
This file implements the model contamination detection through guided prompting.
https://arxiv.org/pdf/2308.08493.pdf
"""

import nltk
import random
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from datasets import Value
from functools import partial
from llmsanitize.utils.utils import seed_everything, fill_template
from llmsanitize.utils.logger import get_child_logger, suspend_logging
from llmsanitize.model_methods.llm import LLM
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
        choices = [_label + '.' + _text for _text, _label in zip(choices['text'], choices['label'])]
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
        raise (f"Error! guided_prompt_split_fn not found processing for dataset_name: {dataset_name}")

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


def filter_data(
    eval_data,
    eval_data_name
):

    data_points = []
    if eval_data_name == "truthful_qa":
        for x in tqdm(eval_data):
            # Remove questions with 4 or less words
            n_words = len(word_tokenize(x["text"]))
            if n_words <= 4:
                continue
            # Remove questions of 'Indexical Error' category
            if 'Indexical Error' in x["category"]:
                continue 

            data_points.append(x)
    else:
        for x in tqdm(eval_data):
            # The other datasets are: {ARC, HellaSwag, MMLU, Winogrande}
            if eval_data_name == "allenai/ai2_arc":
                choices = x["choices"]["text"]
            if eval_data_name == "Rowan/hellaswag":
                choices = x["endings"]
            if eval_data_name == "cais/mmlu":
                choices = x["choices"]
            if eval_data_name == "winogrande":
                choices = [x["option1"], x["option2"]]

            if len(choices) == 2:
                # Remove questions with Yes/No options
                if choices[0].lower() in ["yes", "no"] and choices[1].lower() in ["yes", "no"]:
                    continue
                # Remove questions with True/False options
                if choices[0].lower() in ["true", "false"] and choices[1].lower() in ["true", "false"]:
                    continue

            # Remove data points where the ROUGE-L F1 between any 2 options exceeds 0.65
            scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
            discard = False
            for i in range(len(choices)):
                for j in range(i+1, len(choices)):
                    choice_i = choices[i]
                    choice_j = choices[j]
                    rouge_scores = scorer.score(choice_i, choice_j)
                    rl = rouge_scores["rougeLsum"].fmeasure
                    if rl >= 0.65:
                        discard = True
                        break 
                if discard == True:
                    break
            if discard == True:
                continue

            data_points.append(x)
    logger.info(f"We are left with {len(data_points)} data points")

    return data_points

def main_ts_guessing_question_based(
    eval_data,
    eval_data_name,
    eval_set_key,
    text_key,
    label_key,
    num_proc,
    # model parameters
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
):

    # filter out some data points
    data_points = filter_data(eval_data, eval_data_name)

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

