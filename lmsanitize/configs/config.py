
from pathlib import Path
import yaml
from lmsanitize.utils.utils import dict_to_object

# Read yaml config for variable configs (configuration that can change frequently)
with open(Path(__file__).parent / 'llm_config.yaml', 'r') as rf:
    config_dict = yaml.safe_load(rf)
    config = dict_to_object(config_dict)
    print(config_dict)

# Following are some constants (usually won't change)
FAILURE_TOLERANCE = 2

supported_methods = {
    "gpt-2": "string matching",
    "guided-prompting": "Method in TIME TRAVEL IN LLMS: TRACING DATA CONTAMINATION IN LARGE LANGUAGE MODELS"
}