"""Baseline KNN text classifier."""

import json
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.tools import clean_texts, most_common_words


# Matriz de confusión
def plot_confusion_matrix(y_true, y_pred,k):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig(f"KNN/N{k}_confusion_matrix.png")
    #plt.show()

nltk.download('stopwords')

stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl("dataset_final/train.jsonl")
test_data = load_jsonl("dataset_final/test.jsonl")

X_train_raw = [d["text"] for d in train_data]
y_train = [d["label"] for d in train_data]
X_test_raw = [d["text"] for d in test_data]
y_test = [d["label"] for d in test_data]


# --- Calcular palabras más comunes eliminando stopwords ---
print("Calculando palabras más comunes para eliminación...")
common_words = most_common_words(X_train_raw, stop_words)

custom_stopwords = stop_words.union(set(common_words))

X_train = clean_texts(X_train_raw, custom_stopwords)
X_test = clean_texts(X_test_raw, custom_stopwords)

vectorizer = TfidfVectorizer(max_features=3000, stop_words=list(stop_words))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

for n in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print(f"\nResultados para N={n}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred,n)

