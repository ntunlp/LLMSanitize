import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


# collect all unique n-grams of size ngram_size, streaming over the dataset
def build_ngrams_streaming(data, ngram_size, text_processing_method=None, text_key=None, text_keys=None):
    set_ngrams = {}

    for data_point in tqdm(data):
        if not(text_keys in [[], ['']]):
            text = ""
            for j in range(len(text_keys)):
                key = text_keys[j]
                if j == 0:
                    text += str(data_point[key])
                else:
                    text += " | " + str(data_point[key])
        else:
            text = data_point[text_key]

        clean_text = text
        if text_processing_method != None:
            clean_text = text_processing_method(text)
        ngrams_ = ngrams(sequence=word_tokenize(clean_text), n=ngram_size)
        seen_in_doc = {}
        for ngram in ngrams_:
            if not(ngram in seen_in_doc.keys()):
                if not(ngram in set_ngrams.keys()):
                    set_ngrams[ngram] = 0
                set_ngrams[ngram] += 1
                seen_in_doc[ngram] = 0

    return set_ngrams
