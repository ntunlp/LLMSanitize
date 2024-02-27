import re
import string
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from lmsanitize.configs.config import supported_methods, config
from lmsanitize.utils.method_utils import guided_prompt_process_fn
from lmsanitize.base_contamination_checker import BaseContaminationChecker
from lmsanitize.data_contamination_utils import build_ngrams, tag_ngrams

class DataContaminationChecker(BaseContaminationChecker):
    def __init__(self, args):
        super(DataContaminationChecker, self).__init__(args)

    def run_contamination(self, method):
        if not(method in self.supported_methods.keys()):
            methods = list(self.supported_methods.keys())
            raise KeyError(f'Please pass in a data contamination method which is supported, among: {methods}')

        if method == "gpt-2":
            self.contamination_gpt2()
        elif method == "gpt-3":
            self.contamination_gpt3()

    def contamination_gpt2(self):
        def clean_text_gpt2(text):
            text = text.lower() # lower case
            text = ''.join(i if (i.isalpha() or i==" ") else '' for i in text) # keep alphanumeric characters
            text = re.sub(' +', ' ', text) # only single spaces
            text = text.strip() # initial and final spaces

            return text

        # method-specific dataset processing:
        ## only keep the content per data example, discard labels
        self.train_data = self.train_data[self.text_key]
        self.eval_data = self.eval_data[self.text_key]

        ngram_size = 8
        train_ngrams = build_ngrams(self.train_data, ngram_size, clean_text_gpt2) 
        message = f"There are {len(train_ngrams.keys())} {ngram_size}-grams in the training set"
        print(message)

        all_fracs = tag_ngrams(self.eval_data, train_ngrams, ngram_size, clean_text_gpt2)
        mean_frac = np.mean(all_fracs)
        message = f"{ngram_size}-grams overlap (GPT-2 style data contamination detection) between {self.train_data_name} (train) " \
                  f"and {self.eval_data_name}/{self.eval_set_key}: {mean_frac:.4f}%"
        print(message)

    def contamination_gpt3(self):
        def clean_text_gpt3(text):
            text = text.lower() # lower case
            text = ' '.join(word.strip(string.punctuation) for word in text.split())

            return text
        
        self.train_data = self.train_data[self.text_key]
        self.eval_data = self.eval_data[self.text_key]

        ngram_size = 13
        train_ngrams = build_ngrams(self.train_data, ngram_size, clean_text_gpt3)
        message = f"There are {len(train_ngrams.keys())} {ngram_size}-grams in the training set"
        print(message)

        all_fracs = tag_ngrams(self.eval_data, train_ngrams, ngram_size, clean_text_gpt3)
        mean_frac = np.mean(all_fracs)
        message = f"{ngram_size}-grams overlap (GPT-3 style data contamination detection) between {self.train_data_name} (train) " \
                  f"and {self.eval_data_name}/{self.eval_set_key}: {mean_frac:.4f}%"
        print(message)


