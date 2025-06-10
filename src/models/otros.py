"""Cluster analysis on documents labeled 'Otros'."""

import json
import re
from collections import Counter

import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from src.utils.tools import clean_texts,most_common_words


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))

train_data = load_jsonl("dataset_final/train.jsonl")
test_data = load_jsonl("dataset_final/test.jsonl")

X_train_raw = [d["text"] for d in train_data]
y_train = [d["label"] for d in train_data]
X_test_raw = [d["text"] for d in test_data]
y_test = [d["label"] for d in test_data]


print("Calculando palabras más comunes para eliminación...")
common_words = most_common_words(X_train_raw, stop_words)
custom_stopwords = stop_words.union(set(common_words))



X_train = clean_texts(X_train_raw, custom_stopwords)
X_test = clean_texts(X_test_raw, custom_stopwords)

X_train = [d["text"] for d in train_data]
y_train_raw = [d["label"] for d in train_data]
X_test = [d["text"] for d in test_data]
y_test_raw = [d["label"] for d in test_data]

otros_texts = [x["text"] for x in train_data if x["label"] == "Otros"]
otros_files = [x["label"] for x in train_data if x["label"] == "Otros"]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
otros_embed = model.encode(otros_texts, show_progress_bar=True)

import hdbscan
clust = hdbscan.HDBSCAN(min_cluster_size=5)
otros_clusters = clust.fit_predict(otros_embed)

from collections import defaultdict

grupo_por_cluster = defaultdict(list)
for idx, grupo in enumerate(otros_clusters):
    if grupo != -1:
        grupo_por_cluster[grupo].append((otros_files[idx], otros_texts[idx]))

for grupo, items in grupo_por_cluster.items():
    print(f" Cluster {grupo} ({len(items)} items):")
    for file, text in items[:3]:
        print(text[:150])

