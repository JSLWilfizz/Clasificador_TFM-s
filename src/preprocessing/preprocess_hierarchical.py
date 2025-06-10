import os
import json
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from collections import defaultdict
import nltk
from unidecode import unidecode
from src.utils.tools import most_common_words, clean_texts

# ----- Lista oficial de ODS -----
ODS_LIST = [
    "Fin de la pobreza",
    "Hambre cero",
    "Salud y bienestar",
    "Educación de calidad",
    "Igualdad de género",
    "Agua limpia y saneamiento",
    "Energía asequible y no contaminante",
    "Trabajo decente y crecimiento económico",
    "Industria, innovación e infraestructura",
    "Reducción de las desigualdades",
    "Ciudades y comunidades sostenibles",
    "Producción y consumo responsables",
    "Acción por el clima",
    "Vida submarina",
    "Vida de ecosistemas terrestres",
    "Paz, justicia e instituciones sólidas",
    "Alianzas para lograr los objetivos",
]

# ----- Preprocesamiento y normalización -----
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('spanish') + stopwords.words('english'))

def preprocesar_abstract(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stops and len(w) > 2]
    return ' '.join(tokens)

def normalizar_clase(label):
    label = label.strip()
    if "Ingeniería Civil" in label or "Arquitectura Ingeniería Civil" in label or "Ingeniería Industrial" in label or "Arquitectura" in label:
        return "Ingeniería Civil y Arquitectura"
    elif "Informática" in label or "Telecomunicaciones" in label.split() or "Ingeniería de Sistemas" in label:
        return "Informática y Telecomunicaciones"
    elif "Educación" in label:
        return "Educación"
    else:
        return "Otros"

def get_data(folder):
    texts, labels, labels_2 = [], [], []
    for file in os.listdir(folder):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            abstract = data.get("abstract", "").strip()
            label = data.get("Materias", "").strip()
            label_2 = data.get("ODS", "").strip()
            if abstract and label and label_2:
                norm_label = normalizar_clase(label)
                texts.append(preprocesar_abstract(abstract))
                labels.append(norm_label)
                labels_2.append(label_2)
    df = pd.DataFrame({"text": texts, "label": labels, "label_2": labels_2})
    return df

def extraer_ods_lista(ods_raw, ods_list):
    if not isinstance(ods_raw, str):
        return []
    ods_raw_limpio = re.sub(r'\d+\.\s*', '', ods_raw).strip()
    encontrados = [ods for ods in ods_list if ods in ods_raw_limpio]
    return encontrados

# ---- Cargar y limpiar datos ----
df = get_data("combinados")
df['label_2_list'] = df['label_2'].apply(lambda x: extraer_ods_lista(x, ODS_LIST))
df = df[df['label_2_list'].map(lambda x: len(x) > 0)]
df = df.explode('label_2_list').drop(columns=['label_2']).rename(columns={'label_2_list': 'label_2'}).reset_index(drop=True)
conteo_ods = df['label_2'].value_counts()
ods_validos = conteo_ods[conteo_ods >= 2].index
df = df[df['label_2'].isin(ods_validos)].reset_index(drop=True)

# ---- Split y vectorización ----


X = df['text']
y_materias = df['label']
y_ods = df['label_2']
X_train, X_test, y_materias_train, y_materias_test, y_ods_train, y_ods_test = train_test_split(
    X, y_materias, y_ods, test_size=0.2, random_state=42, stratify=y_materias
)
print("📊 Calculando palabras más comunes para eliminación...")
all_tokens = []
for text in X:
    tokens = re.findall(r"\b\w+\b", text.lower())
    filtered = [t for t in tokens if t not in stops]
    all_tokens.extend(filtered)

most_common_words = most_common_words(X, stops)
custom_stopwords = stops.union(set(most_common_words))

X_train = clean_texts(X_train, custom_stopwords)
X_test = clean_texts(X_test, custom_stopwords)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---- Definir modelos a probar ----
modelos = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": svm.SVC(kernel='linear', C=10, probability=True),
    "Bayes": MultinomialNB(),
    "Arbol": DecisionTreeClassifier(max_depth=10, random_state=42)
}

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

for nombre_modelo_1, modelo_1 in modelos.items():
    print(f"\n\n{'='*20} Primer nivel: {nombre_modelo_1} (Materias) {'='*20}")
    clf_1 = make_pipeline(TfidfVectorizer(max_features=3000), modelo_1)
    clf_1.fit(X_train, y_materias_train)
    y_pred_materias = clf_1.predict(X_test)
    print(classification_report(y_materias_test, y_pred_materias,zero_division=1))

    # Save confusion matrix for Materias
    cm1 = confusion_matrix(y_materias_test, y_pred_materias)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)
    disp1.plot(xticks_rotation=45)
    plt.title(f'Confusion Matrix - {nombre_modelo_1} (Materias)')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/confmat_materias_{nombre_modelo_1}.png')
    plt.close()

    # ---- Segundo nivel: para cada modelo ----
    for nombre_modelo_2, modelo_2 in modelos.items():
        print(f"\n--- Segundo nivel: {nombre_modelo_2} (ODS) ---")
        materia_to_ods_model = defaultdict(lambda: None)
        for materia in df['label'].unique():
            idx = df[df['label'] == materia].index
            X_materia = df.loc[idx, 'text']
            y_materia_ods = df.loc[idx, 'label_2']
            if y_materia_ods.nunique() < 2:
                continue  # Salta materias monoclase
            pipe_2 = make_pipeline(TfidfVectorizer(max_features=3000), modelo_2)
            pipe_2.fit(X_materia, y_materia_ods)
            materia_to_ods_model[materia] = pipe_2

        # Predicción jerárquica
        ods_pred_final = []
        for i, text in enumerate(X_test):
            materia_pred = y_pred_materias[i]
            if materia_pred in materia_to_ods_model and materia_to_ods_model[materia_pred] is not None:
                ods_pred = materia_to_ods_model[materia_pred].predict([text])[0]
            else:
                ods_pred = "desconocido"
            ods_pred_final.append(ods_pred)
        
        print(classification_report(y_ods_test, ods_pred_final,zero_division=1))

        # Save confusion matrix for ODS
        labels_ods = sorted(list(set(list(y_ods_test) + list(ods_pred_final))))
        cm2 = confusion_matrix(y_ods_test, ods_pred_final, labels=labels_ods)
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
        disp2.plot(xticks_rotation=45)
        plt.title(f'Confusion Matrix - {nombre_modelo_1}_{nombre_modelo_2} (ODS)')
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/confmat_ods_{nombre_modelo_1}_{nombre_modelo_2}.png')
        plt.close()
