# GPT-J

import json
import time
from typing import List

import openai
import requests

from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("vllm_utils")


def initialize_post(config):
    """
    initialize openai lib to query local served closed_data
    """
    openai.api_key = "EMPTY"
    openai.api_base = f"http://127.0.0.1:{config.local.port}/v1"

def post_http_request(
    prompt: str,
    api_url: str,
    n: int = 1,
    max_tokens: int = 16,
    temperature: float = 0.0,
    use_beam_search: bool = False,
    stream: bool = False,
    stop: List[str] = ["</s>"],
    echo: bool = False,
    **kwargs
) -> requests.Response:

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

def query_llm_post(config, prompt):
    # Prepare the prompt to the chat template
    # If you are using a chat closed_data, then we recommend using the template.
    if not(config.query.no_chat_template):
        prompt = [{"role": "user", "content": prompt}]
        tokenizer = config.local.tokenizer
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    output_strs = []
    start = time.time()
    engine = {'name': config.openai.model_name}
    response = {}
    while time.time() - start < config.query.max_request_time:
        try:
            response = post_http_request(
                prompt=prompt,
                model=engine['name'],
                api_url=f"http://127.0.0.1:{config.local.port}/v1/completions",
                n=config.query.num_samples,
                max_tokens=config.query.max_tokens,
                temperature=config.query.temperature,  # it's default to 0.0
                use_beam_search=False,
                stream=False,
                echo=config.query.echo,
                logprobs=int(config.query.top_logprobs),
                # Note that vllm openai api uses different parameters of OpenAI's.
                top_logprobs=int(config.query.top_logprobs),
            )
            if response.status_code != 200:
                raise Exception(json.loads(response.content))
            response = json.loads(response.content)
            #print("*"*50)
            #print(prompt)
            #print("*"*10)
            #print(response["choices"][0]["text"])
            output_strs += [
                choice["text"] for choice in response["choices"]
            ]
            break
        except Exception as e:
            logger.info(f"Unexpected exception in generating solution. Sleeping again: {e}")
            time.sleep(config.query.sleep_time)

    if not output_strs:
        output_strs.append('N/A')

    return output_strs, response, 0
