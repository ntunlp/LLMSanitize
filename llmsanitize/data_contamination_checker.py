"""
Contamination detection class for data contamination use cases: func(data1, data2)
"""

import re
import string
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity

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
        elif method =="platypus":
            self.contamination_platypus()

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
        contaminated = np.array([int(x > 0) for x in fractions])
        frac = 100 * np.mean(contaminated)
        n_contaminated = np.sum(contaminated)
        message = f"\nData contamination: checking {self.eval_data_name}/{self.eval_set_key} against {self.train_data_name} (train)"
        message += f"\nMethod: matching of {ngram_size}-grams (GPT-2 style data contamination)"
        message += f"\n# Contaminated points: {n_contaminated}/{len(self.eval_data)} or {frac:.4f}%"
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
        contaminated = np.array([int(x > 0) for x in fractions])
        frac = 100 * np.mean(contaminated)
        n_contaminated = np.sum(contaminated)
        message = f"\nData contamination: checking {self.eval_data_name}/{self.eval_set_key} against {self.train_data_name} (train)"
        message += f"\nMethod: matching of {ngram_size}-grams (GPT-3 style data contamination)"
        message += f"\n# Contaminated points: {n_contaminated}/{len(self.eval_data)} or {frac:.4f}%"
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
        contaminated = overlap_ngrams(self.eval_data, train_ngrams, ngram_size, overlap_thresh, None)
        frac = 100 * np.mean(contaminated)
        n_contaminated = np.sum(contaminated)
        message = f"\nData contamination: checking {self.eval_data_name}/{self.eval_set_key} against {self.train_data_name} (train)"
        message += f"\nMethod: ratio of contaminated {ngram_size}-grams is above {overlap_thresh}% (PaLM style data contamination)"
        message += f"\n# Contaminated points: {n_contaminated}/{len(self.eval_data)} or {frac:.4f}%"
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
        contaminated = fraction_sampled_strings(self.eval_data, train_strings, string_size, n_samples, clean_text_gpt4)
        frac = 100 * np.mean(contaminated)
        n_contaminated = np.sum(contaminated)
        message = f"\nData contamination: checking {self.eval_data_name}/{self.eval_set_key} against {self.train_data_name} (train)"
        message += f"\nMethod: sampling {n_samples} {string_size}-chars substring (GPT-4 style data contamination)"
        message += f"\n# Contaminated points: {n_contaminated}/{len(self.eval_data)} or {frac:.4f}%"
        print(message)

    # Following the logic in Platypus paper: https://arxiv.org/pdf/2308.07317.pdf section 2.2
    def contamination_platypus(self):
        from sentence_transformers import SentenceTransformer

        ## only keep the content per data example, discard labels
        self.train_data = self.train_data["text"]
        self.eval_data = self.eval_data["text"]

        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        train_embeddings = model.encode(self.train_data)
        eval_embeddings = model.encode(self.eval_data)
        cos = cosine_similarity(eval_embeddings, train_embeddings)

        thresh = 0.8
        contaminated = (np.max(cos, axis=1) >= thresh).astype(int)
        frac = 100 * np.mean(contaminated)
        n_contaminated = np.sum(contaminated)
        message = f"\nData contamination: checking {self.eval_data_name}/{self.eval_set_key} against {self.train_data_name} (train)"
        message += f"\nMethod: Sentence-Transformers embeddings cosine above {thresh} (Platypus style)"
        message += f"\n# Contaminated points: {n_contaminated}/{len(self.eval_data)} or {frac:.4f}%"
        print(message)

