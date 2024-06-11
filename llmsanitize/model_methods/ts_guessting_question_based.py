"""
This file implements the model contamination detection through guided prompting.
https://arxiv.org/pdf/2311.09783
"""

import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger
from llmsanitize.utils.logger import get_child_logger, suspend_logging
from llmsanitize.model_methods.llm import LLM

logger = get_child_logger("ts_guessing_question_based")


def build_prompt(text, st):
    tags = st.tag(text.split())
    words = [x for x in tags if x[1] in ['NN', 'JJ', 'VB']]
    idx = np.random.randint(len(words))
    word = words[idx][0]
    for i in range(len(text)-len(word)):
        if text[i:(i+len(word))] == word:
            text = text[:i] + "[MASK]" + text[(i+len(word)):]
            break

    prompt = "Complete the sentence in one word:"
    prompt += f"\n\n{text}"
    prompt += "\nReply the answer only."

    return prompt

def inference(data_points, llm):
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
    responses = []
    for example in tqdm(data_points):
        text = example["text"]
        prompt = build_prompt(text, st)
        response, cost = llm.query(prompt)
        responses.append(response)

    return responses

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

    return data_points

def main_ts_guessing_question_based(
    eval_data,
    eval_data_name,
    n_eval_data_points,
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
    logger.info(f"We are left with {len(data_points)} data points after filtering")

    # perform the shuffling and subsampling now
    if n_eval_data_points > 0:
        p = np.random.permutation(len(data_points))
        data_points = [data_points[x] for x in p]
        data_points = data_points[:n_eval_data_points]
        logger.info(f"After subsampling, there are now {len(data_points)} eval data points")

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

    responses = inference(data_points, llm)
    print(responses[0])
    print(data_points[0])

