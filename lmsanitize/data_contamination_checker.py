import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from lmsanitize.supported_methods import supported_methods


class DataContaminationChecker:
    def __init__(self,
                 train_set="train_set",
                 eval_set="eval_set",
                 eval_key="test",
                 text_key='ctx'):
        self.train_set_name = train_set
        self.eval_set_name = eval_set
        self.eval_key = eval_key
        self.text_key = text_key
        self.supported_methods = supported_methods

        self.download_data()

    def download_data(self):
        self.train_data = load_dataset(self.train_set_name)
        self.train_data = self.train_data['train']
        self.train_data = self.train_data[self.text_key]

        self.eval_data = load_dataset(self.eval_set_name)
        self.eval_data = self.eval_data[self.eval_key]
        self.eval_data = self.eval_data[self.text_key]
        message = f"There are {len(self.train_data)} train elements and {len(self.eval_data)} eval elements"
        print(message)

    def run_contamination(self, method):
        if not(method in self.supported_methods.keys()):
            methods = list(self.supported_methods.keys())
            raise KeyError(f'Please pass in a data contamination method which is supported, among: {methods}')

        if method == "gpt-2":
            self.contamination_gpt2()

    def contamination_gpt2(self):
        def clean_text_gpt2(text):
            text = text.lower() # lower case
            text = ''.join(i if (i.isalpha() or i==" ") else '' for i in text) # keep alphanumeric characters
            text = re.sub(' +', ' ', text) # only single spaces
            text = text.strip() # initial and final spaces

            return text

        train_ngrams = {}
        for i in tqdm(range(len(self.train_data))):
            text_i = self.train_data[i]
            clean_text_i = clean_text_gpt2(text_i)
            ngrams_i = ngrams(sequence=word_tokenize(clean_text_i), n=8)
            for ngram in ngrams_i:
                if not(ngram in train_ngrams.keys()):
                    train_ngrams[ngram] =0
                train_ngrams[ngram] += 1
        message = f"There are {len(train_ngrams.keys())} 8-grams in the training set"
        print(message)

        all_fracs = []
        for i in tqdm(range(len(self.eval_data))):
            text_i = self.eval_data[i]
            clean_text_i = clean_text_gpt2(text_i)
            ngrams_i = ngrams(sequence=word_tokenize(clean_text_i), n=8)
            found, count = 0, 0
            for ngram in ngrams_i:
                if ngram in train_ngrams.keys():
                    found += 1
                count += 1
            frac = 100 * found / count
            all_fracs.append(frac)
        mean_frac = np.mean(all_fracs)
        message = f"8-grams overlap (GPT-2 style data contamination detection) between {self.train_set_name} (train) " \
                  f"and {self.eval_set_name}/{self.eval_key}: {mean_frac:.4f}%"
        print(message)

