import networkx as nx
import matplotlib.pyplot as plt
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
from tools import most_common_words, clean_texts

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

from graphviz import Digraph
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
dot = Digraph(comment='Pipeline Jerárquico Materias → ODS')

dot.attr(rankdir='LR', size='8,5')

dot.node('T', 'Texto (Abstract)')
dot.node('M', 'Modelo Nivel 1:\nClasificación Materias')
dot.node('B', 'Modelo Nivel 2:\nClasificación ODS\n(condicionado a Materia)')
dot.node('O', 'Salida:\nODS predicho')

dot.edges(['TM', 'MB', 'BO'])

# (Opcional: ramas para distintas materias)
for materia in df['label'].unique()[:4]:  # solo algunas para que no se vea enorme
    dot.node(f'ODS_{materia}', f'Modelo ODS para:\n"{materia}"')
    dot.edge('M', f'ODS_{materia}')
    dot.edge(f'ODS_{materia}', 'O')

dot.render('pipeline_jerarquico', view=True)