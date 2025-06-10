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


def normalizar_clase(label):
    label = label.strip().lower()

    # Primero, multidisciplinares: Si el label menciona dos áreas claramente diferenciadas.
    if any(x in label for x in
           ["ingeniería", "industrial", "informática", "mecánica", "eléctrica", "robótica", "tecnología",
            "electrónica"]) and \
            any(x in label for x in
                ["medio ambiente", "energías renovables", "agricultura", "minería", "silvicultura", "alimentos",
                 "botánica", "zoología"]):
        return "Ciencias Ambientales y Agrarias"

    # Arquitectura, Urbanismo y Construcción (palabras muy específicas)
    if any(x in label for x in [
        "arquitectura", "urbanismo", "edificación", "construcción", "obras públicas", "topografía", "caminos"
    ]):
        return "Arquitectura, Urbanismo y Construcción"

    # Ingeniería y Tecnología (si no entró arriba)
    if any(x in label for x in [
        "ingeniería", "informática", "robótica", "electrónica", "telecomunicaciones",
        "mecánica", "optica", "naval", "aeronáutica", "transporte", "industrial", "eléctrica"
    ]):
        return "Ingeniería y Tecnología"

    # Ciencias Naturales y Exactas
    if any(x in label for x in [
        "física", "matemáticas", "química", "biología", "astronomía", "genética", "bioquímica", "geología"
    ]):
        return "Ciencias Naturales y Exactas"

    # Ciencias Sociales y Humanidades
    if any(x in label for x in [
        "economía", "empresa", "psicología", "sociología", "derecho", "filología", "historia",
        "arte", "biblioteconomía", "deportes", "ciencias sociales"
    ]):
        return "Ciencias Sociales y Humanidades"

    # Ciencias Ambientales y Agrarias
    if any(x in label for x in [
        "medio ambiente", "energías renovables", "agricultura", "silvicultura", "botánica", "zoología",
        "minería", "alimentos", "ciencia y tecnología de alimentos", "combustibles fósiles"
    ]):
        return "Ciencias Ambientales y Agrarias"

    # Otros
    return "Otros"


def load_and_filter_data(folder, min_samples=2):
    texts, labels = [], []
    lista_originales = []

    for file in os.listdir(folder):
        if not file.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                # --- CONCATENAR CAMPOS RELEVANTES ---
                partes = []
                if data.get("Título"): partes.append(data["Título"])
                if data.get("Materias"): partes.append(data["Materias"])
                if data.get("Palabras Clave Informales"): partes.append(data["Palabras Clave Informales"])
                if data.get("ODS"): partes.append(data["ODS"])
                if data.get("Departamento"): partes.append(data["Departamento"])
                if data.get("Escuela"): partes.append(data["Escuela"])
                if data.get("abstract"): partes.append(data["abstract"])
                texto_completo = ". ".join(partes)

                label = data.get("Materias", "").strip()
                if texto_completo.strip() and label:
                    norm_label = normalizar_clase(label)
                    texts.append(clean_text(texto_completo))
                    if label not in lista_originales:
                        lista_originales.append(label)
                    labels.append(norm_label)
        except Exception as e:
            print(f"[!] Error procesando {file}: {e}")
            continue

    counts = Counter(labels)
    filtered_texts, filtered_labels = [], []
    for x, y in zip(texts, labels):
        if counts[y] >= min_samples:
            filtered_texts.append(x)
            filtered_labels.append(y)
    print(lista_originales)
    return filtered_texts, filtered_labels

def save_jsonl(file_path, texts, labels):
    with open(file_path, "w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            json.dump({"text": text, "label": label}, f, ensure_ascii=False)
            f.write("\n")

def preparar_y_guardar(data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    X_raw, y_raw = load_and_filter_data(data_folder)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_encoded
    )

    save_jsonl(os.path.join(output_folder, "train.jsonl"), X_train, y_train)
    save_jsonl(os.path.join(output_folder, "test.jsonl"), X_test, y_test)

    with open(os.path.join(output_folder, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(dict(enumerate(label_encoder.classes_)), f, indent=4, ensure_ascii=False)

    print(f"Guardado en: {output_folder}")
    print(f"Clases: {len(label_encoder.classes_)}")
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
                        labels.append(normalizar_clase(label))
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
    plt.title(f"Top {5} clases por frecuencia")
    plt.ylabel("Cantidad de documentos")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preparar_y_guardar("combinados", "dataset_final")
    etiquetas = cargar_etiquetas("combinados")
    plot_distribucion(etiquetas, top_n=5)

