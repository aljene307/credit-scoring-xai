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

Les modules suivants (modélisation, évaluation, explicabilité, API) sont en cours de développement.
"""
)
