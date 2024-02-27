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

    def download_data(self):
        self.train_data = load_dataset(self.train_data_name)
        self.train_data = self.train_data['train']

        self.eval_data = load_dataset(self.eval_data_name)
        self.eval_data = self.eval_data[self.eval_set_key]
        message = f"There are {len(self.train_data)} train elements and {len(self.eval_data)} eval elements"
        print(message)

    def run_contamination(self, method):
        print("run_contamination not implemented")
        pass

