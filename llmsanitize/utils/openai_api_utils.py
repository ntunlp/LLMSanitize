# GPT-J

import json
import time
from typing import List

import openai
import requests

from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("openai_utils")


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
    """
    initialize openai lib to query local served model
    """
    openai.api_key = "EMPTY"
    openai.api_base = f"http://127.0.0.1:{config.local.port}/v1"


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      max_tokens: int = 16,
                      temperature: float = 0.0,
                      use_beam_search: bool = False,
                      stream: bool = False,
                      stop: List[str] = ["</s>"],
                      echo: bool = False,
                      **kwargs) -> requests.Response:
    headers = {"User-Agent": "MERIt Test Client"}
    p_load = {
        "prompt": prompt,
        "n": n,
        "use_beam_search": use_beam_search,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "stop": stop,
        "echo": echo,
    }
    p_load.update(kwargs)
    response = requests.post(api_url, headers=headers, json=p_load, stream=True)
    return response


def query_llm_api(config, prompt):
    output_strs = []
    total_cost = 0
    start = time.time()
    engine = {'name': config.openai.model_name}

    # if prompt is not a list, assign prompt to user_msg
    if type(prompt) == str and not config.local.port:  # Fixed: This should only be used for OpenAI models or vllm chat completions API.
        prompt = [{"role": "user", "content": prompt}]

    response = {}
    while time.time() - start < config.query.max_request_time:
        try:
            if not config.local.port:
                response = openai.ChatCompletion.create(  # Currently `echo` option is deprecated in OpenAI's ChatCompletion API.
                    model=engine['name'],
                    messages=prompt,
                    n=config.query.num_samples,
                    max_tokens=config.query.max_tokens,
                    logprobs=int(config.query.top_logprobs) > 0,  # boolean
                    top_logprobs=int(config.query.top_logprobs),  # int, [0, 5],
                    echo=config.query.echo,
                    temperature=config.query.temperature,
                )

                output_strs += [
                    choice["message"]['content'] for choice in response["choices"]
                ]
                total_cost += calculate_openai_cost(engine['name'], response['usage'])
                break
            else:
                response = post_http_request(
                    prompt=prompt,
                    model=engine['name'],
                    api_url=f"http://127.0.0.1:{config.local.port}/v1/completions",
                    n=config.query.num_samples,
                    max_tokens=config.query.max_tokens,
                    temperature=config.query.temperature, # it's default to 0.0
                    use_beam_search=False,
                    stream=False,
                    echo=config.query.echo,
                    logprobs=int(config.query.top_logprobs),  # Note that vllm openai api uses different parameters of OpenAI's.
                    top_logprobs=int(config.query.top_logprobs),
                )
                if response.status_code != 200:
                    raise Exception(json.loads(response.content))
                response = json.loads(response.content)
                output_strs += [
                    choice["text"] for choice in response["choices"]
                ]
                total_cost += calculate_openai_cost(engine['name'], response['usage'])
                break
        except Exception as e:
            logger.info(f"Unexpected exception in generating solution. Sleeping again: {e}")
            time.sleep(config.query.sleep_time)

    if not output_strs:
        output_strs.append('N/A')

    return output_strs, response, total_cost
