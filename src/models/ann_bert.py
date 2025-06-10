"""Neural network classifier using BERT embeddings."""

import json
import re
from collections import Counter

import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sentence_transformers import SentenceTransformer

from src.utils.tools import clean_texts, most_common_words

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))

train_data = load_jsonl("dataset_final/train.jsonl")
test_data = load_jsonl("dataset_final/test.jsonl")

X_train_raw = [d["text"] for d in train_data]
y_train_raw = [d["label"] for d in train_data]
X_test_raw = [d["text"] for d in test_data]
y_test_raw = [d["label"] for d in test_data]

print("Calculando palabras más comunes para eliminación...")
most_common = most_common_words(X_train_raw, stop_words)
custom_stopwords = stop_words.union(set(most_common))

X_train = clean_texts(X_train_raw, custom_stopwords)
X_test = clean_texts(X_test_raw, custom_stopwords)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)
num_classes = len(label_encoder.classes_)

print("🔄 Generando embeddings con BERT...")
model_bert = SentenceTransformer("distiluse-base-multilingual-cased-v1")
X_train_vec = model_bert.encode(X_train, show_progress_bar=True)
X_test_vec = model_bert.encode(X_test, show_progress_bar=True)

y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_vec.shape[1],)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X_train_vec, y_train_cat, epochs=10, batch_size=32, validation_split=0.1)

y_pred_prob = model.predict(X_test_vec)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_pred)