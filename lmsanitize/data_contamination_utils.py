from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

def build_ngrams(data, ngram_size, text_processing_method=None):
    set_ngrams = {}
    for i in tqdm(range(len(data))):
        text_i = data[i]
        clean_text_i = text_i
        if text_processing_method != None:
            clean_text_i = text_processing_method(text_i)
        ngrams_i = ngrams(sequence=word_tokenize(clean_text_i), n=ngram_size)
        for ngram in ngrams_i:
            if not(ngram in set_ngrams.keys()):
                set_ngrams[ngram] = 0
            set_ngrams[ngram] += 1

    return set_ngrams

def tag_ngrams(data, ngrams_set, ngram_size, text_processing_method=None):
    all_fracs = []
    for i in tqdm(range(len(data))):
        text_i = data[i]
        clean_text_i = text_i
        if text_processing_method != None:
            clean_text_i = text_processing_method(text_i)
        words = word_tokenize(clean_text_i)
        if len(words) < ngram_size:
            continue
        ngrams_i = ngrams(sequence=words, n=ngram_size)
        found, count = 0, 0
        for ngram in ngrams_i:
            if ngram in ngrams_set.keys():
                found += 1
            count += 1
        frac = 100 * found / count
        all_fracs.append(frac)
    
    return all_fracs
