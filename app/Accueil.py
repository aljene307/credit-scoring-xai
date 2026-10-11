"""Point d'entrée de l'application Streamlit : déclare les pages."""

import streamlit as st

st.set_page_config(page_title="Scoring de crédit explicable", layout="wide")

navigation = st.navigation(
    [
        st.Page("views/accueil.py", title="Accueil", default=True),
        st.Page("views/donnees.py", title="1. Données"),
        st.Page("views/preparation.py", title="2. Préparation"),
    ]
)
navigation.run()
