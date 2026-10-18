"""Point d'entrée de l'application Streamlit : déclare les pages."""

import streamlit as st

import credit_scoring.demarrage  # noqa: F401  scikit-learn et scipy chargés une fois, avant les pages

st.set_page_config(page_title="Scoring de crédit explicable", layout="wide")

navigation = st.navigation(
    [
        st.Page("views/accueil.py", title="Accueil", default=True),
        st.Page("views/donnees.py", title="1. Données"),
        st.Page("views/preparation.py", title="2. Préparation"),
        st.Page("views/modelisation.py", title="3. Modélisation"),
        st.Page("views/evaluation.py", title="4. Évaluation"),
        st.Page("views/explicabilite.py", title="5. Explicabilité"),
        st.Page("views/api_demo.py", title="6. API"),
    ]
)
navigation.run()
