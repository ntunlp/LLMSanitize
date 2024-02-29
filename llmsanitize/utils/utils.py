"""
This file includes common utilization functions
"""

import re
import random
import os
import numpy as np
from copy import copy


def clean_train_text(text):
    text = text.lower()
    text = re.sub(r'\W+', '', text)  # keep alphanumeric characters
    text = re.sub(' +', ' ', text)  # only single spaces
    text = text.strip()


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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)


def fill_template(template, vars_map):
    ''' vars_map: {"var_name_in_template": actual_var}
    '''
    new_prompt = copy(template)
    for k, v in vars_map.items():
        new_prompt = new_prompt.replace('{' + k + '}', v)
    return new_prompt
