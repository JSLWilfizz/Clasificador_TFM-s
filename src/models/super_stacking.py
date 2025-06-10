import json
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, recall_score
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
    # Concatenar campos enriquecidos (ajusta según tus campos disponibles)
    partes = []
    if d.get("Título"): partes.append(d["Título"])
    if d.get("Materias"): partes.append(d["Materias"])
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
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
from collections import OrderedDict
import pandas as pd

# ------------------------------
# Define modelos base y meta-modelos
# ------------------------------

BASE_MODELS = OrderedDict({
    "SVM": SVC(kernel='linear', probability=True, class_weight='balanced', C=1, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "NaiveBayes": MultinomialNB(),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42),
    "RF": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "LR": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
})

META_MODELS = OrderedDict({
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42)
})

# Define combinaciones a probar
COMBINACIONES = [
    ["SVM", "RF"],
    ["SVM", "KNN"],
    ["MLP", "RF"],
    ["RF", "SVM"],
    ["MLP", "KNN"],
    ["SVM", "MLP"],
    ["SVM", "RF", "KNN"],
    ["SVM", "RF", "MLP"],
    ["SVM", "KNN", "NaiveBayes"],
    ["KNN", "RF", "NaiveBayes"],
    ["KNN", "MLP", "NaiveBayes"],
    ["SVM", "RF", "LR"],
    ["SVM", "RF", "KNN", "MLP"],
    # Puedes agregar más aquí
]

# Elige qué features usar (TF-IDF, embeddings, etc.)
X_train = X_train_vec
X_test = X_test_vec

resultados = []

print("\n======= COMPARANDO STACKINGS =======")
for base_names in COMBINACIONES:
    for meta_name, meta_model in META_MODELS.items():
        ests = [(name, BASE_MODELS[name]) for name in base_names]
        stacking = StackingClassifier(
            estimators=ests,
            final_estimator=meta_model,
            cv=5,
            n_jobs=1
        )
        try:
            stacking.fit(X_train, y_train)
            y_pred = stacking.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='macro')

            resultados.append({
                "Stacking": "+".join(base_names),
                "Meta": meta_name,
                "Accuracy": round(acc, 3),
                "Macro_F1": round(macro_f1, 3),
                "Weighted_F1": round(weighted_f1, 3),
                "Recall": round(recall, 3)
            })
            print(f"✓ {base_names} / {meta_name} — acc: {acc:.3f}  macro_f1: {macro_f1:.3f} recall: {recall:.3f}")
        except Exception as e:
            print(f"[!] Error con {base_names} / {meta_name}: {e}")

# Modelos individuales para comparación
for name, model in BASE_MODELS.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='macro')
        resultados.append({
            "Stacking": name,
            "Meta": "-",
            "Accuracy": round(acc, 3),
            "Macro_F1": round(macro_f1, 3),
            "Weighted_F1": round(weighted_f1, 3),
            "Recall": round(recall, 3)
        })
        print(f"✓ Modelo individual: {name} — acc: {acc:.3f}  macro_f1: {macro_f1:.3f} recall: {recall:.3f}")
    except Exception as e:
        print(f"[!] Error con modelo individual {name}: {e}")

# Mostrar ranking final
df_res = pd.DataFrame(resultados)
df_res = df_res.sort_values(by=["Macro_F1", "Accuracy"], ascending=False)
print("\n=== RANKING FINAL (mejor macro F1 arriba) ===")
print(df_res.reset_index(drop=True))
df_res.to_csv("resultados_stacking.csv", index=False, encoding='utf-8-sig')

# ---------------------------------
# EVALUACIÓN DETALLADA DEL MEJOR STACKING
# ---------------------------------
# Toma el mejor stacking según macro F1
mejor = df_res.iloc[0]
base_models = mejor['Stacking'].split('+')
meta_name = mejor['Meta']
if meta_name == "-":
    # Es modelo individual
    if len(base_models) == 1:
        best_model = BASE_MODELS[base_models[0]]
    else:
        best_model = None
else:
    ests = [(name, BASE_MODELS[name]) for name in base_models]
    best_model = StackingClassifier(
        estimators=ests,
        final_estimator=META_MODELS[meta_name],
        cv=5,
        n_jobs=1
    )
    best_model.fit(X_train, y_train)

if best_model is not None:
    y_pred = best_model.predict(X_test)
    print("\n=== Classification Report (MEJOR MODELO) ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix (MEJOR MODELO) ===")
    print(confusion_matrix(y_test, y_pred, labels=np.unique(y_test)))
    # Opcional: muestra algunos errores para análisis cualitativo
    errores = np.where(np.array(y_test) != np.array(y_pred))[0]
    for idx in errores[:10]:
        print(f"\nTexto: {X_test_raw[idx][:180]}...")
        print(f"Real: {y_test[idx]}  →  Predicho: {y_pred[idx]}")
        print('-' * 80)

