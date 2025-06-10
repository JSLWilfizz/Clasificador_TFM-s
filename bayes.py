import json
import re
from collections import Counter

import nltk
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
import seaborn as sns

from tools import clean_texts

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
print("📊 Calculando palabras más comunes para eliminación...")
all_tokens = []
for text in X_train_raw:
    tokens = re.findall(r"\b\w+\b", text.lower())
    filtered = [t for t in tokens if t not in stop_words]
    all_tokens.extend(filtered)

freqs = Counter(all_tokens)
most_common_words = [word for word, _ in freqs.most_common(40)]
print("🚫 Palabras adicionales eliminadas:", most_common_words)
custom_stopwords = stop_words.union(set(most_common_words))



X_train = clean_texts(X_train_raw, custom_stopwords)
X_test = clean_texts(X_test_raw, custom_stopwords)

parameters = {
    'alpha': [0.1, 0.5, 1.0],
    'fit_prior': [True, False]
}
model = make_pipeline(TfidfVectorizer(max_features=3000),
                      GridSearchCV(MultinomialNB(), parameters)
)
print("Mejores parámetros encontrados:", model.get_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_pred)
# Displaying a Pipeline with a Preprocessing Step and Regression
from sklearn import set_config
set_config(display="text")