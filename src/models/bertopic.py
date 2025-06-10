"""Topic modeling using BERTopic."""

import json
import re
from collections import Counter

import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from bertopic import BERTopic

from src.utils.tools import clean_texts, most_common_words

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))

# --- Cargar datos ---
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl("dataset_final/train.jsonl")
texts_raw = [d["text"] for d in train_data]
labels_raw = [d["label"] for d in train_data]

# --- Calcular palabras más comunes eliminando stopwords ---
print("Calculando palabras más comunes para eliminación...")
most_common = most_common_words(texts_raw, stop_words)
custom_stopwords = stop_words.union(set(most_common))

texts = clean_texts(texts_raw, custom_stopwords)

print("Ejecutando BERTopic...")
topic_model = BERTopic(language="multilingual")
topics, probs = topic_model.fit_transform(texts)

topic_model.visualize_topics().show()
topic_model.visualize_barchart(top_n_topics=10).show()
topic_model.visualize_hierarchy().show()
print(topic_model.get_topic_info())
with open("bertopic_result.csv", "w", encoding="utf-8") as f:
    f.write("file,label,topic\n")
    for d, label, topic in zip(train_data, labels_raw, topics):
        f.write(f"{d.get('file', '')},{label},{topic}\n")

print("Resultados guardados en bertopic_result.csv")
