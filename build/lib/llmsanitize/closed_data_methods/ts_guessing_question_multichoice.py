"""
This file implements the closed_data contamination detection through guided prompting.
https://arxiv.org/pdf/2311.09783
"""

import os
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize, sent_tokenize
from llmsanitize.utils.logger import get_child_logger
from llmsanitize.utils.dataset_utils import get_answers_list, get_answer_index
from llmsanitize.closed_data_methods.llm import LLM
from llmsanitize.closed_data_methods.ts_guessing_question_based import filter_data

logger = get_child_logger("ts_guessing_question_multichoice")


def build_prompt(
    example,
    eval_data_name
):
    text = example["text"]

    choices = get_answers_list(example, eval_data_name)
    answer_index = get_answer_index(example, eval_data_name)
    answer = choices[answer_index]
    wrong_choices_indices = [i for i in range(len(choices)) if i != answer_index]
    index = np.random.randint(len(wrong_choices_indices))
    wrong_choice_index = wrong_choices_indices[index]

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    wrong_letter = alphabet[wrong_choice_index]
    prompt = f"Please fill in the [] in option {wrong_letter} based on your benchmark knowledge."
    prompt += "\n\nThe crucial rule is that you should provide different answer in other options below."
    prompt += f"\n\nQuestion: {text}"
    prompt += "\nOptions:"
    for i in range(len(choices)):
        letter = alphabet[i]
        if i == wrong_choice_index:
            prompt += f"\n{letter}: [MASK]"
        else:
            choice = choices[i]
            prompt += f"\n{letter}: [{choice}]"
    prompt += "\n\nReply with answer only."

    return prompt, answer, wrong_letter

def process_response(response, wrong_letter):
    symbol = wrong_letter + ":"
    if symbol in response:
        response = response.split(symbol)[1]
        sents = sent_tokenize(response)
        if len(sents) > 0:
            response = sents[0]

    return response

def inference(
    data_points,
    eval_data_name,
    llm
):
    responses, answers = [], []
    for example in tqdm(data_points):
        prompt, answer, wrong_letter = build_prompt(
            example,
            eval_data_name
        )
        response, cost = llm.query(prompt)
        response = process_response(response, wrong_letter)
        responses.append(response)
        answers.append(answer)

    return responses, answers

def main_ts_guessing_question_multichoice(
    eval_data,
    eval_data_name,
    n_eval_data_points,
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
):
    # filter out some open_data points
    data_points = filter_data(eval_data, eval_data_name)
    logger.info(f"We are left with {len(data_points)} data points after filtering")

    # perform the shuffling and subsampling now
    if n_eval_data_points > 0:
        p = np.random.permutation(len(data_points))
        data_points = [data_points[x] for x in p]
        data_points = data_points[:n_eval_data_points]
        logger.info(f"We are left with {len(data_points)} data points after subsampling")

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

    responses, answers = inference(
        data_points,
        eval_data_name,
        llm
    )

    responses = [x.lower() for x in responses]
    answers = [x.lower() for x in answers]
    print("HERE")
    print(responses[0])
    print(answers[0])
    em = len([i for i in range(len(responses)) if responses[i] == answers[i]]) / len(responses)
    scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    rl = np.mean(np.array([scorer.score(responses[i], answers[i])["rougeLsum"].fmeasure for i in range(len(responses))]))

    logger.info(f"Question-based guessing")
    logger.info(f"Exact Match (EM): {em:.2f}, ROUGE-L F1: {rl:.2f}")
