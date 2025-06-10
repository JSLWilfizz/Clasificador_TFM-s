"""HDBSCAN clustering of text embeddings."""

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

from src.utils.tools import clean_texts, most_common_words

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
common_words = most_common_words(texts_raw, stop_words)
custom_stopwords = stop_words.union(set(common_words))



texts = clean_texts(texts_raw, custom_stopwords)

print("Generando embeddings con BERT")
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
X_embed = model.encode(texts, show_progress_bar=True)

print("Ejecutando HDBSCAN...")
hdb = hdbscan.HDBSCAN(gen_min_span_tree=True)
y_hdb = hdb.fit_predict(X_embed)
#Resultados
print("Número de clusters encontrados:", len(set(y_hdb)) - (1 if -1 in y_hdb else 0))
print("Adjusted Rand Index:", adjusted_rand_score(y_true, y_hdb))
print("Normalized Mutual Information:", normalized_mutual_info_score(y_true, y_hdb))


# --- Visualización con PCA ---
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_embed)

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_hdb, cmap='tab10', s=10)
plt.title("Clusters generados con HDBSCAN + BERT")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# --- Exportar CSV con cluster y label ---
print("Exportando resultados a hdbscan_result.csv")
with open("resultados/hdbscan_result.csv", "w", encoding="utf-8") as f:
    f.write("file,label,cluster\n")
    for d, label, cluster in zip(train_data, y_true, y_hdb):
        f.write(f"{d.get('file', '')},{label},{cluster}\n")
