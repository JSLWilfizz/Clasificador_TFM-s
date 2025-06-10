import os
import json
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_filter_data(folder, min_samples=2):
    texts, labels = [], []

    for file in os.listdir(folder):
        if not file.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                abstract = data.get("abstract", "").strip()
                label = data.get("Materias", "").strip()
                if abstract and label:
                    texts.append(clean_text(abstract))
                    labels.append(label)
        except:
            continue

    # Filtrar clases raras
    counts = Counter(labels)
    filtered_texts, filtered_labels = [], []
    for x, y in zip(texts, labels):
        if counts[y] >= min_samples:
            filtered_texts.append(x)
            filtered_labels.append(y)

    return filtered_texts, filtered_labels

def save_jsonl(file_path, texts, labels):
    with open(file_path, "w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            json.dump({"text": text, "label": int(label)}, f, ensure_ascii=False)
            f.write("\n")

def preparar_y_guardar(data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    X_raw, y_raw = load_and_filter_data(data_folder)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Guardar archivos
    save_jsonl(os.path.join(output_folder, "train.jsonl"), X_train, y_train)
    save_jsonl(os.path.join(output_folder, "test.jsonl"), X_test, y_test)

    # Guardar el mapeo de clases
    with open(os.path.join(output_folder, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(dict(enumerate(label_encoder.classes_)), f, indent=4, ensure_ascii=False)

    print(f"✅ Guardado en: {output_folder}")
    print(f"🎯 Clases: {len(label_encoder.classes_)}")
    print(label_encoder.classes_)
def cargar_etiquetas(folder_path):
    labels = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            path = os.path.join(folder_path, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    label = data.get("Materias", "").strip()
                    if label:
                        labels.append(label)
            except:
                pass
    return labels

def plot_distribucion(labels, top_n=20):
    counter = Counter(labels)
    más_comunes = counter.most_common(top_n)
    clases, conteos = zip(*más_comunes)

    plt.figure(figsize=(12, 6))
    plt.bar(clases, conteos)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} clases por frecuencia")
    plt.ylabel("Cantidad de documentos")
    plt.tight_layout()
    plt.show()


# Ejecutar
if __name__ == "__main__":
    preparar_y_guardar("combinados", "dataset_final")
    etiquetas = cargar_etiquetas("combinados")
    plot_distribucion(etiquetas, top_n=28)

