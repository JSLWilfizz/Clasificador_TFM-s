import json
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------------
# UTILIDADES
# ------------------------------

def clean_text(text):
    # Limpieza estándar (puedes ajustarla)
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def cargar_jsonl(path):
    # Carga archivos .jsonl a listas
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preparar_texto_registro(d):
    # Concatenar campos enriquecidos
    partes = []
    if d.get("Título"): partes.append(d["Título"])
    if d.get("Palabras Clave Informales"): partes.append(d["Palabras Clave Informales"])
    if d.get("ODS"): partes.append(d["ODS"])
    if d.get("Departamento"): partes.append(d["Departamento"])
    if d.get("Escuela"): partes.append(d["Escuela"])
    if d.get("text"): partes.append(d["text"])      # Compatibilidad
    if d.get("abstract"): partes.append(d["abstract"])
    texto = ". ".join([str(p) for p in partes if p])
    return clean_text(texto)

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión (Stacking)")
    plt.tight_layout()
    plt.show()

# ------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ------------------------------

# Rutas (ajusta según tu estructura)
train_path = "dataset_final/train.jsonl"
test_path = "dataset_final/test.jsonl"

train_data = cargar_jsonl(train_path)
test_data = cargar_jsonl(test_path)

X_train_raw = [preparar_texto_registro(d) for d in train_data]
y_train = [d["label"] for d in train_data]
X_test_raw = [preparar_texto_registro(d) for d in test_data]
y_test = [d["label"] for d in test_data]

print("Ejemplo texto enriquecido:", X_train_raw[0][:300])
print("Clases (train):", Counter(y_train))

# ------------------------------
# VECTORIZACIÓN
# ------------------------------

vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train_raw)
X_test_vec = vectorizer.transform(X_test_raw)

# ------------------------------
# STACKING CLASSIFIER
# ------------------------------

# Modelos base
estimators = [
    ('svm', SVC(kernel='linear', probability=True, class_weight='balanced', C=1, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42),
    cv=5,
    n_jobs=1
)

print("Entrenando stacking...")
stacking.fit(X_train_vec, y_train)

# ------------------------------
# EVALUACIÓN
# ------------------------------

y_pred = stacking.predict(X_test_vec)
print("\n=== Classification Report (Stacking) ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix (Stacking) ===")
print(confusion_matrix(y_test, y_pred, labels=stacking.classes_))

plot_confusion_matrix(y_test, y_pred, labels=stacking.classes_)

#muestra algunos errores para análisis cualitativo
errores = np.where(np.array(y_test) != np.array(y_pred))[0]
for idx in errores[:10]:
    print(f"\nTexto: {X_test_raw[idx][:180]}...")
    print(f"Real: {y_test[idx]}  →  Predicho: {y_pred[idx]}")
    print('-' * 80)
