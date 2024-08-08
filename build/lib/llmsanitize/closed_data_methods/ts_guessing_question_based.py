"""
This file implements the closed_data contamination detection through guided prompting.
https://arxiv.org/pdf/2311.09783
"""

import os
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger
from llmsanitize.utils.logger import get_child_logger, suspend_logging
from llmsanitize.utils.dataset_utils import get_answers_list
from llmsanitize.closed_data_methods.llm import LLM

logger = get_child_logger("ts_guessing_question_based")


def get_stanford_tagger():
    if not("CLASSPATH" in os.environ and "STANFORD_MODELS" in os.environ):
        logger.info("You are using a closed_data contamination detection method which requires Stanford's Part-of-Speech tagger.")
        logger.info("You need to setup global variables CLASSPATH and STANFORD_MODELS specifying the path to the tagger.")
        logger.info("First download the tagger here: https://nlp.stanford.edu/software/tagger.html#Download")
        logger.info("Then place it into some directory.")
        home_dir = input("Please specify the directory where you place the tagger (default: /home/mathieu/stanford-postagger-full-2020-11-17): ")
        os.environ["CLASSPATH"] = f"{home_dir}"
        os.environ["STANFORD_MODELS"] = f"{home_dir}/models"
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')

    return st

def build_prompt(
    example, 
    tagger,
    eval_data_name,
    type_hint=False,
    category_hint=False,
    url_hint=False
):
    text = example["text"]
    tags = tagger.tag(text.split())
    words = [x for x in tags if x[1] in ['NN', 'JJ', 'VB']]
    if len(words) == 0:
        return "failed", ""
    idx = np.random.randint(len(words))
    word = words[idx][0]
    for i in range(len(text)-len(word)+1):
        if text[i:(i+len(word))] == word:
            text = text[:i] + "[MASK]" + text[(i+len(word)):]
            break

    prompt = "Complete the sentence in one word:"
    prompt += f"\n\n{text}"
    if type_hint:
        if eval_data_name == "truthful_qa":
            example_type = example["type"]
            prompt += f"\nHint: {example_type}"
    if category_hint:
        if eval_data_name == "truthful_qa":
            example_category = example["category"]
            prompt += f"\nHint: {example_category}"
    if url_hint:
        if eval_data_name == "truthful_qa":
            example_url = example["source"]
            prompt += f"\nHint: {example_url}"
    prompt += "\nReply the answer only."

    return prompt, word

def process_response(response):
    processed_response = word_tokenize(response)[0]

    return processed_response

def inference(
    data_points, 
    n_eval, 
    eval_data_name,
    llm, 
    type_hint=False,
    category_hint=False,
    url_hint=False
):
    tagger = get_stanford_tagger()

    responses, masked_words = [], []
    for example in tqdm(data_points):
        prompt, masked_word = build_prompt(
            example, 
            tagger,
            eval_data_name,
            type_hint,
            category_hint,
            url_hint
        )
        if prompt == "failed":
            continue
        response, cost = llm.query(prompt)
        response = process_response(response)
        responses.append(response)
        masked_words.append(masked_word)
        if len(responses) == n_eval:
            break

    return responses, masked_words

@suspend_logging
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
        scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
        for x in tqdm(eval_data):
            choices = get_answers_list(x, eval_data_name)

            if len(choices) == 2:
                # Remove questions with Yes/No options
                if (choices[0].lower() in ["yes", "no"]) and (choices[1].lower() in ["yes", "no"]):
                    continue
                # Remove questions with True/False options
                if (choices[0].lower() in ["true", "false"]) and (choices[1].lower() in ["true", "false"]):
                    continue

            # Remove open_data points where the ROUGE-L F1 between any 2 options exceeds 0.65
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

    return data_points

def main_ts_guessing_question_based(
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
    # method-specific parameters
    type_hint: bool = False,
    category_hint: bool = False,
    url_hint: bool = False,
):
    # filter out some open_data points
    data_points = filter_data(eval_data, eval_data_name)
    logger.info(f"We are left with {len(data_points)} data points after filtering")

    # perform the shuffling and subsampling now
    if n_eval_data_points > 0:
        p = np.random.permutation(len(data_points))
        data_points = [data_points[x] for x in p]

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

    responses, masked_words = inference(
        data_points, 
        n_eval_data_points,
        eval_data_name,
        llm, 
        type_hint,
        category_hint,
        url_hint
    )
    
    responses = [x.lower() for x in responses]
    masked_words = [x.lower() for x in masked_words]
    em = len([i for i in range(len(responses)) if responses[i] == masked_words[i]]) / len(responses)
    logger.info(f"Question-based completion (type hint: {type_hint} | category hint: {category_hint} | url hint: {url_hint})")
    logger.info(f"Exact Match (EM): {em:.2f}")
