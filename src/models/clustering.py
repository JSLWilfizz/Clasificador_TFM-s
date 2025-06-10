"""Agglomerative clustering using BERT embeddings."""

import json
import nltk
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sentence_transformers import SentenceTransformer
import re

from src.utils.tools import clean_texts, most_common_words

nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')
english_stopwords = stopwords.words('english')
stop_words = set(spanish_stopwords + english_stopwords)

# --- Cargar datos ---
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl("dataset_final/train.jsonl")
texts_raw = [d["text"] for d in train_data]
y_true = [d["label"] for d in train_data]

# --- Contar palabras más frecuentes eliminando stopwords ---
print("Calculando palabras más comunes...")
common_words = most_common_words(texts_raw, stop_words)
print("Palabras adicionales a eliminar:", common_words)

# --- Eliminar stopwords + palabras más frecuentes ---
combined_stopwords = stop_words.union(common_words)
texts = clean_texts(texts_raw, combined_stopwords)

# --- Embeddings con BERT ---
print("Generando embeddings con BERT multilingüe...")
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
X_embed = model.encode(texts, show_progress_bar=True)

plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(X_embed, method ='ward')),no_labels=True)


print("Ejecutando Agglomerative Clustering...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_true)
num_labels = len(set(y_encoded))

hclust = AgglomerativeClustering(n_clusters=2)
y_hclust = hclust.fit_predict(X_embed)

# --- Resultados ---
print("Número de clusters encontrados:", len(set(y_hclust)) - (1 if -1 in y_hclust else 0))
print("\nEvaluación del clustering vs etiquetas reales")
print("Rand Index:", adjusted_rand_score(y_encoded, y_hclust))
print("Normalized Mutual Info (NMI):", normalized_mutual_info_score(y_encoded, y_hclust))

# --- Visualización con PCA ---
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_embed)




plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_hclust, cmap='tab10', s=10)
plt.title("Clusters con Agglomerative Clustering + BERT")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# --- Exportar CSV con cluster y label ---
print("Exportando resultados a hclust_result.csv")
with open("resultados/hclust_result.csv", "w", encoding="utf-8") as f:
    f.write("file,label,cluster\n")
    for d, label, cluster in zip(train_data, y_true, y_hclust):
        f.write(f"{d.get('file', '')},{label},{cluster}\n")
