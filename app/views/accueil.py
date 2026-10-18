"""Page d'accueil."""

import streamlit as st

st.title("Scoring de crédit explicable")
st.markdown(
    """
Cette application prédit le **défaut de paiement** d'un client de carte de crédit,
compare plusieurs modèles et **explique** leurs décisions, globalement et client par client.

**Données :** *Default of Credit Card Clients* (Taiwan, 2005), UCI Machine Learning Repository,
licence CC BY 4.0. 30 000 clients, 23 variables. Le jeu est téléchargé automatiquement ;
vous pouvez aussi importer votre propre fichier CSV.

**Modules :**
1. **Données** : aperçu, statistiques, classes, graphiques exploratoires.
2. **Préparation** : choix des variables, séparation entraînement / test, prétraitement.
3. **Modélisation** : quatre modèles réglés par GridSearchCV, tableau comparatif.
4. **Évaluation** : métriques sur le jeu de test, courbes ROC et précision-rappel, seuil, calibration.
5. **Explicabilité** : importance des variables, SHAP, LIME, explication d'un client, équité.
6. **API** : le service FastAPI qui prédit et explique un client, appelé par la page 5.
"""
)
