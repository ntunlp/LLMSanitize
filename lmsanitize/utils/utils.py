"""
This file includes common utilization functions
"""

import re

def clean_train_text(text):
    text = text.lower()
    text = re.sub(r'\W+', '', text) # keep alphanumeric characters
    text = re.sub(' +', ' ', text) # only single spaces
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