import requests
from bs4 import BeautifulSoup
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter, Retry

BASE_URL = "https://oa.upm.es"
PDF_DIR = "pdfs"
META_DIR = "metadata"
MAX_ID = 100000
THREADS = 16

# Reusable session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))


def init_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)


def get_html(url):
    try:
        r = session.get(url, timeout=5)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except Exception:
        return ""


def extract_table_data(url):
    html = get_html(url)
    if not html:
        return None, None, None

    soup = BeautifulSoup(html, 'html.parser')
    table_data = {}
    pdf_url = None

    table = soup.find('table', class_='abstract_description')
    if table:
        rows = table.find_all('tr')
        for row in rows:
            header = row.find('th')
            data = row.find('td')
            if header and data:
                key = header.get_text(strip=True).replace(':', '')
                value = ' '.join(data.get_text().split())
                table_data[key] = value

    pdf_link = soup.find('a', class_='ep_document_link')
    if pdf_link and pdf_link['href'].endswith('.pdf'):
        href = pdf_link['href']
        pdf_url = href if href.startswith("http") else BASE_URL + href

    return table_data, table_data.get("Tipo de Documento"), pdf_url


def save_metadata(file_id, metadata):
    path = os.path.join(META_DIR, f"{file_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def download_pdf(pdf_url, file_id):
    path = os.path.join(PDF_DIR, f"{file_id}.pdf")
    try:
        r = session.get(pdf_url, stream=True, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
    except Exception:
        print(f"[!] Error descargando PDF {file_id}")


def process_id(file_id):
    url = f"{BASE_URL}/{file_id}/"
    table_data, type_value, pdf_url = extract_table_data(url)

    if type_value == "Tesis (Master)":
        print(f"[+] {file_id} - Tesis (Master)")
        save_metadata(file_id, table_data)
        download_pdf(pdf_url, file_id)
    else:
        print(f"[-] {file_id} - Ignorado")


def get_last_id():
    if not os.path.exists(META_DIR):
        return 500000
    files = [f for f in os.listdir(META_DIR) if f.endswith(".json")]
    if not files:
        return 500000
    return max(int(f.split('.')[0]) for f in files) + 1


if __name__ == '__main__':
    init_dirs()
    start_id = get_last_id()
    print(start_id)
    ids = range(start_id, MAX_ID)

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(process_id, i) for i in ids]
        for f in as_completed(futures):
            _ = f.result()
