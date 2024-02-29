import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from llmsanitize.configs.config import supported_methods, config

class BaseContaminationChecker:
    """ Base class of ContaminationChecker
    """
    def __init__(self, args):
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.supported_methods = supported_methods
        self.download_data()

        # standardize the text field
        if self.text_keys != []:
            self.combine_text_keys()
        else:
            self.normalize_text_key()

    def download_data(self):
        self.train_data = load_dataset(self.train_data_name)
        self.train_data = self.train_data['train']

        self.eval_data = load_dataset(self.eval_data_name)
        self.eval_data = self.eval_data[self.eval_set_key]
        message = f"There are {len(self.train_data)} train elements and {len(self.eval_data)} eval elements"
        print(message)

    def combine_text_keys(self):
        for key in self.text_keys:
            assert key in self.train_data.features, "Error - please provide a text key that is in this dataset"
        self.train_data = self.combine_text_keys_subset_(self.train_data)
        self.eval_data = self.combine_text_keys_subset_(self.eval_data)

    def combine_text_keys_subset_(self, subset):
        texts = []
        vals = {}
        for key in self.text_keys:
            vals[key] = subset[key]
        for i in tqdm(range(len(subset))):
            text = ""
            for j in range(len(self.text_keys)):
                key = self.text_keys[j]
                if j == 0:
                    text += vals[key][i]
                else:
                    text += " | " + vals[key][i]
            texts.append(text)
        subset = subset.add_column("text", texts)
        
        return subset

    def normalize_text_key(self):
        self.train_data = self.normalize_text_key_(self.train_data)
        self.eval_data = self.normalize_text_key_(self.eval_data)

    def normalize_text_key_(self, subset):
        assert self.text_key in subset.features, "Error - please provide a text key that is in this dataset"
        subset = subset.add_column("text", subset[self.text_key])
        
        return subset

    def run_contamination(self, method):
        print("run_contamination not implemented")
        pass

