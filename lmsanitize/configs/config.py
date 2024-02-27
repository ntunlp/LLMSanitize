from pathlib import Path
import yaml
from lmsanitize.utils.utils import dict_to_object
from dataclasses import dataclass
import torch

# Read yaml config for variable configs (configuration that can change frequently)
with open(Path(__file__).parent / 'main_config.yaml', 'r') as rf:
    config_dict = yaml.safe_load(rf)
    config = dict_to_object(config_dict)
    print(config_dict)

supported_methods = {dic['name']: dic for dic in config_dict['methods']}

# Following are some constants (usually won't change)
FAILURE_TOLERANCE = 2


# Commented by Fangkai: We need a config class here to specify all default values of parameters and hide them from the config file,
#  so that the user does not feel it complex.
@dataclass
class QueryArguments:
    model_name: str = None
    num_samples: int = 1
    max_tokens: int = 128
    top_logprobs: int = 0
    max_request_time: int = 0
    sleep_time: int = 0


@dataclass
class LocalModelArguments:
    port: str = None
    model_path: str = None
    tokenizer_path: str = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SanitizeArguments:
    # openai
    creds_key_file: str = None
    # inference by local model
    local: LocalModelArguments = LocalModelArguments()
    # query parameters
    query: QueryArguments = QueryArguments()

# TODO: Add initialization for the argument classes above from the yaml config file.
