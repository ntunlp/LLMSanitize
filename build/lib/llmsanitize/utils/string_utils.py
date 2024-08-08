import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


# collect all unique n-grams of size ngram_size
def build_ngrams(data, ngram_size, text_processing_method=None):
    set_ngrams = {}
    for i in tqdm(range(len(data))):
        doc_i = data[i]
        clean_text_i = doc_i
        if text_processing_method != None:
            clean_text_i = text_processing_method(doc_i)
        ngrams_i = ngrams(sequence=word_tokenize(clean_text_i), n=ngram_size)
        seen_in_doc = {}
        for ngram in ngrams_i:
            if not(ngram in seen_in_doc.keys()):
                if not(ngram in set_ngrams.keys()):
                    set_ngrams[ngram] = 0
                set_ngrams[ngram] += 1
                seen_in_doc[ngram] = 0

    return set_ngrams

# find ngrams in eval dataset which have been seen before
def overlap_ngrams(data, ngrams_set, ngram_size, text_processing_method=None):
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
        overlap = (found, count)
        overlaps.append(overlap)

    return overlaps

# collect all full string samples
def build_full_strings(data, text_processing_method=None):
    set_strings = {}
    for i in tqdm(range(len(data))):
        doc_i = data[i]
        clean_text_i = doc_i
        if text_processing_method != None:
            clean_text_i = text_processing_method(doc_i)

        set_strings[clean_text_i] = 0

    return set_strings

# collect all unique strings of size string_size
def build_substrings(data, string_size, text_processing_method=None):
    set_strings = {}
    for i in tqdm(range(len(data))):
        text_i = data[i]
        clean_text_i = text_i
        if text_processing_method != None:
            clean_text_i = text_processing_method(text_i)

        if len(clean_text_i) < string_size:
            strings_i = {clean_text_i: 0}
        else:
            strings_i = {}
            for j in range(len(clean_text_i)-string_size):
                string = clean_text_i[j:(j+string_size)]
                strings_i[string] = 0
        for string in strings_i.keys():
            if not(string in set_strings.keys()):
                set_strings[string] = 0
            set_strings[string] += 1

    return set_strings

# compute the fraction of strings which have been seen before
def overlap_substrings_sample(data, strings_set, string_size, n_samples, text_processing_method=None):
    all_tagged = []
    for i in tqdm(range(len(data))):
        text_i = data[i]
        clean_text_i = text_i
        if text_processing_method != None:
            clean_text_i = text_processing_method(text_i)

        tagged = 0
        if len(clean_text_i) <= string_size:
            for k in strings_set.keys():
                if k.startswith(clean_text_i):
                    tagged = 1
                    break
        else:
            for _ in range(n_samples):
                start_idx = np.random.randint(0, len(clean_text_i)-string_size, 1)[0]
                string = clean_text_i[start_idx:(start_idx+string_size)]
                if string in strings_set.keys():
                    tagged = 1
                    break
        all_tagged.append(int(tagged > 0))

    return all_tagged
