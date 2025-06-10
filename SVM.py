import json
import nltk
import spacy
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import re
import seaborn as sns

# Descargar recursos necesarios
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Cargar modelo spaCy para español
nlp_es = spacy.load("es_core_news_md")

stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))
lemmatizer_en = WordNetLemmatizer()

# Palabras "vacías" específicas del dominio académico
domain_stopwords = set([
    "tesis", "trabajo", "fin", "master", "máster", "presente", "realizado", "desarrollado",
    "universidad", "alumno", "autor", "tutor", "departamento", "investigación"
])

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def detect_language(text):
    return bool(re.search(r'[áéíóúñ]', text.lower()))

def extract_keywords(text):
    # Puedes hacer una versión más compleja con YAKE o similar
    # Aquí simplemente seleccionamos las palabras más largas y menos frecuentes
    words = re.findall(r'\b\w{7,}\b', text.lower())
    return list(set(words) - stop_words)

def lemmatize_and_clean(text):
    # Solo baja a minúsculas y quita signos de puntuación
    text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', text.lower())
    # Detecta idioma
    if detect_language(text):
        doc = nlp_es(text)
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]
    else:
        tokens = text.split()
        tokens = [lemmatizer_en.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def combinar_campos(d):
    if "text" in d and d["text"]:
        return d["text"]
    else:
        return ""

train_data = load_jsonl("dataset_final/train.jsonl")
test_data = load_jsonl("dataset_final/test.jsonl")


X_train_raw = [combinar_campos(d) for d in train_data]
y_train = [d["label"] for d in train_data]
X_test_raw = [combinar_campos(d) for d in test_data]
y_test = [d["label"] for d in test_data]


print("Ejemplo antes de limpiar:", X_train_raw[0][:200])
# Limpieza y lematización
X_train_clean = [lemmatize_and_clean(x) for x in X_train_raw]
X_test_clean = [lemmatize_and_clean(x) for x in X_test_raw]

# Quita clases con < N ejemplos
min_examples = 15
df_train = list(zip(X_train_clean, y_train))
class_counts = Counter(y_train)
valid_classes = {c for c, n in class_counts.items() if n >= min_examples}
df_train = [x for x in df_train if x[1] in valid_classes]
X_train_clean, y_train = zip(*df_train)

# Vectorización: bigramas, stopwords, sin palabras de baja frecuencia
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    stop_words=list(stop_words)
)

for x in X_train_clean[:5]:
    print("Limpio:", x)

X_train_vec = vectorizer.fit_transform(X_train_clean)
X_test_vec = vectorizer.transform(X_test_clean)

# Undersampling (balancea a la clase minoritaria)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train_vec, y_train)
print("Distribución tras undersampling:", Counter(y_resampled))

# Ajuste automático de folds de CV
min_class_size = min(Counter(y_resampled).values())
cv_folds = min(5, min_class_size) if min_class_size >= 2 else 2

# Entrenamiento SVM
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 15]}
model = svm.SVC(class_weight='balanced')
clf = GridSearchCV(model, parameters, cv=cv_folds)
clf.fit(X_resampled, y_resampled)
print("Mejores parámetros encontrados:", clf.best_params_)
y_pred = clf.predict(X_test_vec)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusi\u00f3n")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_pred)

# Muestra ejemplos de errores para análisis cualitativo
import numpy as np
errors = np.where(np.array(y_test) != np.array(y_pred))[0]
for idx in errors[:10]:
    print(f"\nTexto: {X_test_clean[idx][:180]}...")
    print(f"Real: {y_test[idx]}  →  Predicho: {y_pred[idx]}")
    print('-' * 80)
