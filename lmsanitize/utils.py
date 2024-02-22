import re

def clean_train_text(text):
    text = text.lower()
    text = re.sub(r'\W+', '', text) # keep alphanumeric characters
    text = re.sub(' +', ' ', text) # only single spaces
    text = text.strip()