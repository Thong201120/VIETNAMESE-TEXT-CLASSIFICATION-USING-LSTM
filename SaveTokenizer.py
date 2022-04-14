import io
import json
import pickle
import sqlite3

from keras.models import model_from_json
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from pyvi.ViTokenizer import ViTokenizer
import string

df = pd.read_csv('data.csv')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['data'].values)

tokenizer_json = tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))