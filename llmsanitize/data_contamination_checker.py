import re
import string
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from llmsanitize.configs.config import supported_methods, config
from llmsanitize.utils.method_utils import guided_prompt_process_fn
from llmsanitize.base_contamination_checker import BaseContaminationChecker
from llmsanitize.utils.string_utils import *

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
        elif method == "palm":
            self.contamination_palm()
        elif method == "gpt-4":
            self.contamination_gpt4()

    # Following the logic in GPT-2's paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf section 4
    def contamination_gpt2(self):
        # method-specific dataset processing:
        def clean_text_gpt2(text):
            text = text.lower() # lower case
            text = ''.join(i if (i.isalpha() or i==" ") else '' for i in text) # keep alphanumeric characters
            text = re.sub(' +', ' ', text) # only single spaces
            text = text.strip() # initial and final spaces

            return text

        ## only keep the content per data example, discard labels
        self.train_data = self.train_data["text"]
        self.eval_data = self.eval_data["text"]

        ngram_size = 8
        train_ngrams = build_ngrams(self.train_data, ngram_size, clean_text_gpt2) 
        message = f"There are {len(train_ngrams.keys())} {ngram_size}-grams in the training set"
        print(message)

        fractions = fraction_ngrams(self.eval_data, train_ngrams, ngram_size, clean_text_gpt2)
        mean_frac = np.mean(fractions)
        message = f"{ngram_size}-grams overlap (GPT-2 style data contamination detection) between {self.train_data_name} (train) " \
                  f"and {self.eval_data_name}/{self.eval_set_key}: {mean_frac:.4f}%"
        print(message)

    # Following the logic in GPT-3's paper: https://arxiv.org/pdf/2005.14165.pdf section C
    def contamination_gpt3(self):
        # method-specific dataset processing:
        def clean_text_gpt3(text):
            text = text.lower() # lower case
            text = ' '.join(word.strip(string.punctuation) for word in text.split())

            return text
        
        self.train_data = self.train_data["text"]
        self.eval_data = self.eval_data["text"]

        ngram_size = 13
        train_ngrams = build_ngrams(self.train_data, ngram_size, clean_text_gpt3)
        message = f"There are {len(train_ngrams.keys())} {ngram_size}-grams in the training set"
        print(message)

        fractions = fraction_ngrams(self.eval_data, train_ngrams, ngram_size, clean_text_gpt3)
        mean_frac = np.mean(fractions)
        message = f"{ngram_size}-grams overlap (GPT-3 style data contamination detection) between {self.train_data_name} (train) " \
                  f"and {self.eval_data_name}/{self.eval_set_key}: {mean_frac:.4f}%"
        print(message)

    # Following the logic in PaLM's paper: https://arxiv.org/pdf/2204.02311.pdf section 8
    def contamination_palm(self):
        ## only keep the content per data example, discard labels
        self.train_data = self.train_data["text"]
        self.eval_data = self.eval_data["text"]

        ngram_size = 8
        train_ngrams = build_ngrams(self.train_data, ngram_size, None)
        message = f"There are {len(train_ngrams.keys())} {ngram_size}-grams strings in the training set"
        print(message)

        overlap_thresh = 70
        overlaps = overlap_ngrams(self.eval_data, train_ngrams, ngram_size, overlap_thresh, None)
        frac_overlap = 100 * np.mean(overlaps)
        message = f"Ratio of data points with more than 70% 8-grams overlap (PaLM style data contamination detection) between {self.train_data_name} (train) " \
                  f"and {self.eval_data_name}/{self.eval_set_key}: {frac_overlap:.4f}%"
        print(message)

    # Following the logic in GPT-4's report: https://arxiv.org/pdf/2303.08774.pdf appendix C
    def contamination_gpt4(self):
        # method-specific dataset processing:
        def clean_text_gpt4(text):
            text = ''.join(i if i.isalpha() else '' for i in text) # keep alphanumeric characters

            return text

        ## only keep the content per data example, discard labels
        self.train_data = self.train_data["text"]
        self.eval_data = self.eval_data["text"]

        string_size = 50
        train_strings = build_strings(self.train_data, string_size, clean_text_gpt4)
        message = f"There are {len(train_strings.keys())} {string_size}-chars strings in the training set"
        print(message)

        n_samples = 3
        fractions = fraction_sampled_strings(self.eval_data, train_strings, string_size, n_samples, clean_text_gpt4)
        mean_frac = 100 * np.mean(fractions)
        message = f"50-chars string matching ratio (GPT-4 style data contamination detection) between {self.train_data_name} (train) " \
                  f"and {self.eval_data_name}/{self.eval_set_key}: {mean_frac:.4f}%"
        print(message)
