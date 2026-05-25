import requests
from bs4 import BeautifulSoup
import pdfplumber
import re

def clean(txt: str):
    if not txt:
        return "" 

    else:
        txt = re.sub(r'\s+', ' ', txt)
        text = re.sub(r'[^a-zA-Z0-9áàâäãåçéèêëíìîïñóòôöõúùûüýÿæœÁÀÂÄÃÅÇÉÈÊËÍÌÎÏÑÓÒÔÖÕÚÙÛÜÝŸÆŒ\s]', '', text)
    return text.strip().lower()


def scrape_url(url: str):
    """Extrait le texte principal d'une page Web."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Supprimer les scripts et les styles
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.decompose()
            return clean(soup.get_text())
        else:
            return f"Erreur HTTP: {response.status_code}"
    except Exception as e:
        return f"Erreur lors du scraping de l'URL: {str(e)}" 


def extract_pdf(file_like_object):
    """Extrait le texte d'un fichier PDF."""
    try:
        with pdfplumber.open(file_like_object) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return clean(text)
    except Exception as e:
        return f"Erreur lors du scraping du PDF: {str(e)}"