import json
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, adjusted_rand_score, \
    normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import seaborn as sns
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re
import hdbscan

from tools import clean_texts

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))

# --- Cargar datos ---
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl("dataset_final/train.jsonl")
texts_raw = [d["text"] for d in train_data]
y_true = [d["label"] for d in train_data]

# --- Calcular palabras más comunes eliminando stopwords ---
print("Calculando palabras más comunes para eliminación...")
all_tokens = []
for text in texts_raw:
    tokens = re.findall(r"\b\w+\b", text.lower())
    filtered = [t for t in tokens if t not in stop_words]
    all_tokens.extend(filtered)

freqs = Counter(all_tokens)
most_common_words = [word for word, _ in freqs.most_common(30)]
print("🚫 Palabras adicionales eliminadas:", most_common_words)
custom_stopwords = stop_words.union(set(most_common_words))



texts = clean_texts(texts_raw, custom_stopwords)

print("Generando embeddings con BERT")
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
X_embed = model.encode(texts, show_progress_bar=True)

from denseclus import DenseClus

import logging # to further silence deprecation warnings

logging.captureWarnings(True)
clf = DenseClus(
    random_state=1,
    umap_combine_method="intersection_union_mapper"
)
df = pd.DataFrame(X_embed)

clf.fit(X_embed)

embedding = clf.mapper_.embedding_
labels = clf.score()
clustered = (labels >= 0)

cnts = pd.DataFrame(labels)[0].value_counts()
cnts = cnts.reset_index()
cnts.columns = ['cluster','count']
print(cnts.sort_values(['cluster']))
print("Número de clusters encontrados:", len(set(labels)) - (1 if -1 in labels else 0))
