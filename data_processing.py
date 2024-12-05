import pandas as pd
import numpy as np
from IPython.display import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import re
import matplotlib.pyplot as plt
import string

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def ingestData():
    df = pd.read_csv("/Users/prupro/Desktop/Github/NewsClassRNN/artifacts/bbc-news-data.csv", sep="\t")
    df = df[((~df.title.isnull()) & (~df.content.isnull()))].reset_index(drop=True)
    print(f'Data shape and size is : {df.shape} and {df.shape[0]}')
    return df

def data_cleaning(text):

    # Lower the words in the sentence
    cleaned = text.lower()
    # Replace the full stop with a full stop and space
    cleaned = cleaned.replace(".", ". ")
    # Remove the stop words : optional pre-processing step
    tokens = [word for word in cleaned.split() if not word in stop_words]
    # Remove the punctuations
    tokens = [tok.translate(str.maketrans(' ', ' ', string.punctuation)) for tok in tokens]
    # Joining the tokens back to form the sentence
    cleaned = " ".join(tokens)
    # Remove any extra spaces
    cleaned = cleaned.strip()

    return cleaned

def set_max_seq_length():
    max_sentence_len = df['title'].str.split(" ").str.len().max()
    total_classes = df.category.nunique()
    print(f"Maximum sequence length: {max_sentence_len}")
    print(f"Total classes: {total_classes}")

if __name__ == "__main__":
    df = ingestData()
    for index, data in tqdm(df.iterrows(), total=df.shape[0]):
        df.loc[index, 'title'] = data_cleaning(data['title'])
    set_max_seq_length()