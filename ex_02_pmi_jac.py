import re
from glob import glob
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import nltk
from nltk.collocations import *


def clean_dx_bullet(dx):
    res = re.findall(r'^\d+\s?\.\)?\s?(.*)', dx, flags=re.I)
    if res:
        return res[0]
    res = re.findall(r'^\s?-\s?(.*)', dx, flags=re.I)
    if res:
        return res[0]
    return dx


def number_of_letters(x):
    return len([item for item in x if item.isalpha()])


dataset = []
for file in glob(r'data/docs/*.txt'):
    text = None
    with open(file) as f:
        text = f.read()
    if text:
        past_medical_history = re.findall(r'^past medical history(.*?)(?:social\shistory|family\shistory|allergies|'
                                          r'medications\son\sadmission|home\smedications|medications\son\stransfer|'
                                          r'physical\sexam|brief\shospital\scourse)',
                                          text,
                                          flags=re.I | re.S | re.MULTILINE)
        # 630 medical diagnoses in 89 medical charts
        if past_medical_history:
            for past_dx in past_medical_history[0].split('\n'):
                past_dx = clean_dx_bullet(past_dx)
                if len(past_dx.strip()) > 1:
                    dataset.append(past_dx)

words = []
for x in dataset:
    # only accept words that contain at least 1 letter.
    words.extend([item for item in x.lower().split() if number_of_letters(item) > 1])

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words)

print(f'mutual information: {finder.nbest(bigram_measures.mi_like, 20)}')
print('===========================================')
print(f'jaccard: {finder.nbest(bigram_measures.jaccard, 20)}')