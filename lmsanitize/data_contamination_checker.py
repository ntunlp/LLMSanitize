import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from lmsanitize.configs.config import supported_methods, config
from lmsanitize.utils.method_utils import guided_prompt_process_fn
from lmsanitize.base_contamination_checker import BaseContaminationChecker

class DataContaminationChecker(BaseContaminationChecker):
    def __init__(self, args):
        super(DataContaminationChecker, self).__init__(args)

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

        # method-specific dataset processing:
        ## only keep the content per data example, discard labels
        self.train_data = self.train_data[self.text_key]
        self.eval_data = self.eval_data[self.text_key]

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
        message = f"8-grams overlap (GPT-2 style data contamination detection) between {self.train_data_name} (train) " \
                  f"and {self.eval_data_name}/{self.eval_set_key}: {mean_frac:.4f}%"
        print(message)
