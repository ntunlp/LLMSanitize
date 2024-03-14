# GPT-J
import openai
import time
import random
from pathlib import Path


def calculate_openai_cost(engine_name, usage_dict):
    pricing_json = {
        "gpt-3.5-turbo": [0.001, 0.002],
        "gpt-3.5-turbo-0125": [0.0005, 0.0015],
        "gpt-3.5-turbo-1106": [0.001, 0.002],
        "gpt-3.5-turbo-0613": [0.001, 0.002],
        "gpt-4-0125-preview": [0.01, 0.03],
        "gpt-4-1106-preview": [0.01, 0.03],
        "gpt-4": [0.03, 0.06],
        "gpt-4-32k": [0.06, 0.12],

    }
    if engine_name not in pricing_json.keys():
        return 0
    input_price, output_price = pricing_json[engine_name]
    return input_price * (usage_dict['prompt_tokens'] / 1000.) + output_price * (usage_dict['completion_tokens'] / 1000.)


def initialize_openai(config):
    with open(config.openai.creds_key_file, 'r') as rf:
        api_key = rf.read()
    openai.api_key = api_key


def initialize_openai_local(config):
    ''' initialize openai lib to query local served model
    '''
    openai.api_key = "EMPTY"
    openai.api_base = f"http://127.0.0.1:{config.local.port}/v1"


def query_llm_api(config, prompt):
    output_strs = []
    total_cost = 0
    start = time.time()
    engine = {'name': config.openai.model_name}

    # if prompt is not a list, assign prompt to user_msg
    if type(prompt) == str:
        prompt = [{"role": "user", "content": prompt}]

    response = {}
    while time.time() - start < config.query.max_request_time:
        try:
            response = openai.ChatCompletion.create(
                model=engine['name'],
                messages=prompt,
                n=config.query.num_samples,
                max_tokens=config.query.max_tokens,
                logprobs=int(config.query.top_logprobs) > 0,  # boolean
                top_logprobs=int(config.query.top_logprobs),  # int, [0, 5]
            )

            output_strs += [
                choice["message"]['content'] for choice in response["choices"]  # TODO: The response keys should be checked.
            ]
            total_cost += calculate_openai_cost(engine['name'], response['usage'])
            break
        except Exception as e:
            print(
                f"Unexpected exception in generating solution. Sleeping again: {e}"
            )
            time.sleep(config.query.sleep_time)

    if not output_strs:
        output_strs.append('N/A')

    return output_strs, response, total_cost
