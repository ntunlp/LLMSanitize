import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# collect all unique n-grams of size ngram_size
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

# compute the fraction of n-grams which have been seen before
def fraction_ngrams(data, ngrams_set, ngram_size, text_processing_method=None):
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

# check the fraction of ngrams overlap
def overlap_ngrams(data, ngrams_set, ngram_size, overlap_thresh, text_processing_method=None):
    overlaps = []
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
        overlap = int(frac > overlap_thresh)
        overlaps.append(overlap)

    return overlaps

# collect all unique strings of size string_size
def build_strings(data, string_size, text_processing_method=None):
    set_strings = {}
    for i in tqdm(range(len(data))):
        text_i = data[i]
        clean_text_i = text_i
        if text_processing_method != None:
            clean_text_i = text_processing_method(text_i)
        if len(clean_text_i) < string_size:
            strings = [clean_text_i]
        else:
            strings = []
            for j in range(len(clean_text_i)-string_size):
                string = clean_text_i[j:(j+string_size)]
                strings.append(string)
        for string in strings:
            if not(string in set_strings.keys()):
                set_strings[string] = 0
            set_strings[string] += 1

    return set_strings

# compute the fraction of strings which have been seen before
def fraction_sampled_strings(data, strings_set, string_size, n_samples, text_processing_method=None):
    all_tagged = []
    for i in tqdm(range(len(data))):
        text_i = data[i]
        clean_text_i = text_i
        if text_processing_method != None:
            clean_text_i = text_processing_method(text_i)
        if len(clean_text_i) <= string_size:
            continue

        tagged = 0
        for _ in range(n_samples):
            start_idx = np.random.randint(0, len(clean_text_i)-string_size, 1)[0]
            string = clean_text_i[start_idx:(start_idx+string_size)]
            if string in strings_set.keys():
                tagged += 1
        all_tagged.append(int(tagged > 0))

    return all_tagged