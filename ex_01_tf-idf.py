import re
from glob import glob
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer


def clean_dx_bullet(dx):
    res = re.findall(r'^\d+\s?\.\)?\s?(.*)', dx, flags=re.I)
    if res:
        return res[0]
    res = re.findall(r'^\s?-\s?(.*)', dx, flags=re.I)
    if res:
        return res[0]
    return dx


dataset = []
for file in glob(r'data/docs/*.txt'):
    text = None
    with open(file) as f:
        text = f.read()
    if text:
        past_medical_history = re.findall(r'^past medical history(.*?)(?:social\shistory|family\shistory|allergies|'
                                          r'medications\son\sadmission|home\smedications|medications\son\stransfer|'
                                          r'physical\sexam|brief\shospital\scourse|medications)',
                                          text,
                                          flags=re.I | re.S | re.MULTILINE)
        # 630 medical diagnoses in 89 medical charts
        if past_medical_history:
            for past_dx in past_medical_history[0].split('\n'):
                past_dx = clean_dx_bullet(past_dx)
                if len(past_dx.strip()) > 1:
                    dataset.append(past_dx)


tfIdfVectorizer = TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(dataset)

for i in range(len(dataset)):
    print(dataset[i])
    df = pd.DataFrame(tfIdf[i].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    print(df.head(5))
    print('===============================')





