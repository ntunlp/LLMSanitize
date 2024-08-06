import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


# collect all unique n-grams of size ngram_size, streaming over the dataset
def build_ngrams_streaming(data, ngram_size, text_processing_method=None, text_key=None, text_keys=None):
    set_ngrams = {}
    for data_point in tqdm(data):
        text = combine_text_streaming(data_point, text_key=text_key, text_keys=text_keys)
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

# collect all full string samples, streaming over the dataset
def build_full_strings_streaming(data, text_processing_method=None, text_key=None, text_keys=None):
    set_strings = {}
    for data_point in tqdm(data):
        text = combine_text_streaming(data_point, text_key=text_key, text_keys=text_keys)
        clean_text = text
        if text_processing_method != None:
            clean_text = text_processing_method(text)

        set_strings[clean_text] = 0

    return set_strings

# collect all unique strings of size string_size
def build_substrings_streaming(data, string_size, text_processing_method=None, text_key=None, text_keys=None):
    set_strings = {}
    for data_point in tqdm(data):
        text = combine_text_streaming(data_point, text_key=text_key, text_keys=text_keys)
        clean_text = text
        if text_processing_method != None:
            clean_text = text_processing_method(text)

        if len(clean_text) < string_size:
            strings_ = {clean_text: 0}
        else:
            strings_ = {}
            for j in range(len(clean_text)-string_size):
                string = clean_text[j:(j+string_size)]
                strings_[string] = 0
        for string in strings_.keys():
            if not(string in set_strings.keys()):
                set_strings[string] = 0
            set_strings[string] += 1

    return set_strings

# collect embeddings from specified closed_data
def build_embeddings_streaming(data, model, bufer_size=10000, text_processing_method=None, text_key=None, text_keys=None):
    set_embeddings = []
    current_texts = []
    for data_point in tqdm(data):
        text = combine_text_streaming(data_point, text_key=text_key, text_keys=text_keys)
        clean_text = text
        if text_processing_method != None:
            clean_text = text_processing_method(text)
        current_texts.append(clean_text)

        if len(current_texts) >= bufer_size:
            current_embeddings = model.encode(current_texts)
            set_embeddings.append(current_embeddings)
            current_texts = []
    if len(current_texts) > 0:
        current_embeddings = model.encode(current_texts)
        set_embeddings.append(current_embeddings)
    set_embeddings = np.concatenate(set_embeddings)

    return set_embeddings

def combine_text_streaming(data_point, text_key=None, text_keys=None):
    if not (text_keys in [[], ['']]):
        text = ""
        for j in range(len(text_keys)):
            key = text_keys[j]
            if j == 0:
                text += str(data_point[key])
            else:
                text += " | " + str(data_point[key])
    else:
        text = data_point[text_key]

    return text
