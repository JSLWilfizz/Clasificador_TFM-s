import re
from collections import Counter


def clean_texts(texts, extra_stopwords):
    # --- Limpiar textos eliminando stopwords adicionales ---
    cleaned = []
    for text in texts:
        tokens = re.findall(r"\b\w+\b", text.lower())
        filtered = [t for t in tokens if t not in extra_stopwords]
        cleaned.append(" ".join(filtered))
    return cleaned


def most_common_words(X_train_raw, stop_words):
    # --- Calcular palabras más comunes eliminando stopwords ---
    print("Calculando palabras más comunes para eliminación...")
    all_tokens = []
    for text in X_train_raw:
        tokens = re.findall(r"\b\w+\b", text.lower())
        filtered = [t for t in tokens if t not in stop_words]
        all_tokens.extend(filtered)

    freqs = Counter(all_tokens)
    most_common_words = [word for word, _ in freqs.most_common(30)]
    print("Palabras adicionales eliminadas:", most_common_words)
    return most_common_words