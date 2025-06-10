import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))

# Cargar un texto para ejemplo
with open("dataset_final/train.jsonl", "r", encoding="utf-8") as f:
    example = json.loads(f.readlines()[3])  # primer ejemplo
    text = example["text"]

# Tokenizar
vectorizer = TfidfVectorizer(max_features=2000, stop_words=list(stop_words))
X_train = vectorizer.fit_transform([text])

# Convertir a DataFrame
df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out()).T
df.columns = ["TF-IDF"]
df = df[df["TF-IDF"] > 0].sort_values("TF-IDF", ascending=False)

print("📌 Texto original:")
print(text[:1500], "...")  # Mostrar solo los primeros X caracteres
print("\n🔍 Tokens con mayor peso TF-IDF:")
print(df.head(15))