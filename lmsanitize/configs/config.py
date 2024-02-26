
from pathlib import Path
import yaml
from lmsanitize.utils.utils import dict_to_object

# Read yaml config for variable configs (configuration that can change frequently)
with open(Path(__file__).parent / 'main_config.yaml', 'r') as rf:
    config_dict = yaml.safe_load(rf)
    config = dict_to_object(config_dict)
    print(config_dict)

supported_methods = {dic['name']:dic for dic in config_dict['methods']}

# Following are some constants (usually won't change)
FAILURE_TOLERANCE = 2

