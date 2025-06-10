import os
import json
import PyPDF2
import re

def extract_abstract(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except:
                    return None

            text = ""
            for page in reader.pages[:50]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            # Búsqueda del encabezado
            header_pattern = re.compile(r"\b(abstract|resumen|summary)\b", re.IGNORECASE)
            match = header_pattern.search(text)

            if not match:
                return None

            start_idx = match.end()  # posición justo después del encabezado
            chunk = text[start_idx:start_idx + 6000]  # leer 2000 caracteres desde ahí

            # Buscar posibles fin de sección
            end_match = re.search(
                r"\b(keywords|palabras clave|introduction|introducción|1\.\s+introduction)\b",
                chunk,
                re.IGNORECASE
            )
            abstract = chunk[:end_match.start()] if end_match else chunk

            abstract = abstract.strip()

            # Filtro: evitar textos demasiado cortos o de índice
            if len(abstract) < 100 or re.search(r"\.{4,}", abstract):
                return None

            return abstract
    except Exception as e:
        return f"ERROR::{str(e)}"

def combinar_datos(pdf_folder, metadata_folder, output_folder, log_file):
    os.makedirs(output_folder, exist_ok=True)
    sin_abstracts = []

    for pdf_file in os.listdir(pdf_folder):
        if not pdf_file.endswith(".pdf"):
            continue

        base_name = os.path.splitext(pdf_file)[0]
        pdf_path = os.path.join(pdf_folder, pdf_file)
        metadata_path = os.path.join(metadata_folder, base_name + ".json")

        if not os.path.exists(metadata_path):
            print(f"Metadatos no encontrados para {pdf_file}")
            continue

        try:
            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
        except Exception as e:
            print(f"Error leyendo metadatos de {metadata_path}: {str(e)}")
            continue

        abstract = extract_abstract(pdf_path)

        if abstract is None or (isinstance(abstract, str) and abstract.startswith("ERROR::")):
            sin_abstracts.append({"file": pdf_file, "error": abstract or "No se encontró abstract"})
            continue

        combinado = metadata.copy()
        combinado["abstract"] = abstract
        combinado["file"] = pdf_file

        output_path = os.path.join(output_folder, base_name + ".json")
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(combinado, out_file, indent=4, ensure_ascii=False)
        print(f"Combinado guardado: {output_path}")

    # Guardar log
    if sin_abstracts:
        with open(log_file, "w", encoding="utf-8") as log:
            for entry in sin_abstracts:
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nSe registraron {len(sin_abstracts)} archivos sin abstract en: {log_file}")


if __name__ == "__main__":
    combinar_datos("pdfs", "metadata", "combinados", "sin_abstracts.txt")
