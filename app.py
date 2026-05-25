import streamlit as st
import pandas as pd
from scraper import scrape_url, extract_pdf
from model_utils import predict_document

st.set_page_config(page_title="Classification de Documents Médicaux", layout="wide")

st.title("📋 Classification Automatisée de Documents (Médical vs Non-Médical)")
st.write("Insérez une URL ou téléversez un document PDF pour scraper son contenu, le classifier et exporter les résultats.")

# Initialisation de la session state pour stocker l'historique de la databank locale
if 'databank' not in st.session_state:
    st.session_state.databank = []

# --- SECTION INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌐 Option 1 : Entrer une URL")
    url_input = st.text_input("Lien du document à analyser :", placeholder="https://example.com/article")
    submit_url = st.button("Scraper et Analyser l'URL")

with col2:
    st.subheader("📄 Option 2 : Importer un fichier PDF")
    pdf_file = st.file_uploader("Glissez votre fichier ici :", type=["pdf"])
    submit_pdf = st.button("Analyser le PDF")

# --- TRAITEMENT DES FLUX ---
extracted_text = ""
source_name = ""

if submit_url and url_input:
    with st.spinner("Scraping de la page web en cours..."):
        extracted_text = scrape_url(url_input)
        source_name = url_input
elif submit_pdf and pdf_file:
    with st.spinner("Extraction du texte du PDF en cours..."):
        extracted_text = extract_pdf(pdf_file)
        source_name = pdf_file.name

# --- CLASSIFICATION ET AJOUT À LA DATABANK ---
if extracted_text:
    if "Erreur" in extracted_text:
        st.error(extracted_text)
    elif len(extracted_text.strip()) < 10:
        st.warning("Le contenu extrait est trop court pour être analysé de manière fiable.")
    else:
        # Inférence de l'IA
        classe, confiance = predict_document(extracted_text)
        
        # Affichage du résultat immédiat
        st.success(f"Analyse réussie pour : **{source_name}**")
        c1, c2 = st.columns(2)
        c1.metric(label="Classification", value=classe)
        c2.metric(label="Indice de confiance", value=f"{confiance} %")
        
        # Ajout à l'historique (La Databank)
        st.session_state.databank.append({
            "Source": source_name,
            "Contenu Épuré": extracted_text[:200] + "...", # Version tronquée pour lisibilité dans le tableau
            "Classification": classe,
            "Confiance (%)": confiance
        })

# --- VISUALISATION ET EXPORT ---
st.write("---")
st.subheader("📊 Databank Globale (Historique de la session)")

if st.session_state.databank:
    df_result = pd.DataFrame(st.session_state.databank)
    st.dataframe(df_result, use_container_width=True)
    
    # Conversion du DataFrame complet en CSV pour le téléchargement
    csv_data = df_result.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Télécharger la Databank au format CSV",
        data=csv_data,
        file_name="databank_classification_medicale.csv",
        mime="text/csv"
    )
else:
    st.info("Aucun document n'a encore été traité. Les résultats apparaîtront ici sous forme de tableau exportable.")