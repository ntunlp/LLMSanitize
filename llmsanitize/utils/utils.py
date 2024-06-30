"""
This file includes common utilization functions
"""

import re
import random
import os
import torch
import numpy as np
from copy import copy


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def dict_to_object(dict_):
    class Struct(object):
        def __init__(self, data):
            for name, value in data.items():
                setattr(self, name, self._wrap(value))

        def _wrap(self, value):
            if isinstance(value, (tuple, list, set, frozenset)):
                return type(value)([self._wrap(v) for v in value])
            else:
                return Struct(value) if isinstance(value, dict) else value

    return Struct(dict_)

def fill_template(template, vars_map):
    ''' vars_map: {"var_name_in_template": actual_var}
    '''
    new_prompt = copy(template)
    for k, v in vars_map.items():
        new_prompt = new_prompt.replace('{' + k + '}', v)
    return new_prompt
