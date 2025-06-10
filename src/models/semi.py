"""Semi-supervised label spreading using BERT embeddings."""

import json
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelSpreading
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import seaborn as sns
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re

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

# --- Calcular palabras más frecuentes eliminando stopwords ---
common_words = most_common_words(texts_raw, stop_words)
custom_stopwords = stop_words.union(set(common_words))


texts = clean_texts(texts_raw, custom_stopwords)

# --- Embeddings con BERT ---
print("🔄 Generando embeddings con BERT multilingüe...")
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
X_embed = model.encode(texts, show_progress_bar=True)

# --- Codificar etiquetas y ocultar parte (semi-supervisado) ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_true)

# Ocultar 80% de etiquetas para simular aprendizaje semi-supervisado
rng = np.random.default_rng(seed=42)
mask = rng.random(len(y_encoded)) < 0.2
semi_labels = np.copy(y_encoded)
semi_labels[~mask] = -1  # etiquetas desconocidas

# --- Modelo semi-supervisado: Label Spreading ---
print(" Entrenando LabelSpreading semi-supervisado...")
label_model = LabelSpreading(kernel="rbf", alpha=0.2)
label_model.fit(X_embed, semi_labels)
preds = label_model.transduction_

# --- Evaluación del modelo ---
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
print("\n📊 Evaluación del semi-supervisado con etiquetas reales")
print("🔁 Rand Index:", adjusted_rand_score(y_encoded, preds))
print("🔁 Normalized Mutual Info (NMI):", normalized_mutual_info_score(y_encoded, preds))
print("\nClassification Report:")
print(classification_report(y_encoded, preds, target_names=label_encoder.classes_))

# --- Visualización con PCA ---
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_embed)

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=preds, cmap='tab10', s=10)
plt.title("Clusters inferidos por LabelSpreading")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Etiqueta Inferida")
plt.tight_layout()
plt.show()

# --- Exportar CSV con predicciones ---
print("💾 Exportando resultados a semi_supervised_result.csv")
with open("resultados/semi_supervised_result.csv", "w", encoding="utf-8") as f:
    f.write("file,label,predicted\n")
    for d, label, pred in zip(train_data, y_true, preds):
        f.write(f"{d.get('file', '')},{label},{label_encoder.inverse_transform([pred])[0]}\n")