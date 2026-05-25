import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def train_initial_model():
    """Entraîne un modèle de base si aucun modèle existant n'est trouvé."""
    # Simulation de données d'entraînement (Idéalement, enrichir ce dictionnaire)
    data = {
        'text': [
            "Le paracétamol est indiqué en cas de douleur et de fièvre chez l'adulte et l'enfant.",
            "Le patient présente des symptômes de toux sèche, fièvre et insuffisance respiratoire aiguë.",
            "La chirurgie cardiaque consiste à opérer le cœur ou les gros vaisseaux thoraciques.",
            "Le développement web en Python utilise des frameworks comme Flask, Django ou FastAPI.",
            "Le match de football de ce soir a été reporté à cause de la pluie battante.",
            "Déploiement d'une infrastructure cloud sécurisée sur AWS avec Terraform et Docker."
        ],
        'label': ["Medical", "Medical", "Medical", "Non-Medical", "Non-Medical", "Non-Medical"]
    }
    
    df = pd.DataFrame(data)
    
    vectorizer = TfidfVectorizer(stop_words='english') # Remplacer par une liste française si besoin
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Sauvegarde
    with open(MODEL_PATH, 'wb') as m_file:
        pickle.dump(model, m_file)
    with open(VECTORIZER_PATH, 'wb') as v_file:
        pickle.dump(vectorizer, v_file)

def predict_document(text):
    """Prédit si un texte est Médical ou Non-Médical et retourne la probabilité."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        train_initial_model()
        
    with open(MODEL_PATH, 'rb') as m_file:
        model = pickle.load(m_file)
    with open(VECTORIZER_PATH, 'rb') as v_file:
        vectorizer = pickle.load(v_file)
        
    X_text = vectorizer.transform([text])
    prediction = model.predict(X_text)[0]
    probabilities = model.predict_proba(X_text)[0]
    
    # Récupérer l'index de la classe prédite pour obtenir son score de confiance
    class_idx = list(model.classes_).index(prediction)
    confidence = probabilities[class_idx] * 100
    
    return prediction, round(confidence, 2)