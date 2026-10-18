"""Module 6 : démonstration de l'API FastAPI (état, modèles, requêtes et réponses brutes)."""

import pandas as pd
import streamlit as st

from credit_scoring import api_client, config
from credit_scoring.reference import load_reference, reference_split

MODELES = {"complet": "XGBoost complet (23 variables)", "sans_sensibles": "XGBoost sans variables sensibles"}
COMMANDE = ".venv\\Scripts\\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000"


@st.cache_resource(show_spinner="Chargement du jeu de test…")
def test_clients(path: str, modified: float) -> pd.DataFrame:
    results, _ = load_reference()
    return reference_split(results["meta"])[1]


st.title("6. API d'interprétation")
st.markdown(
    "L'API est un **service séparé** : un programme qui reçoit des demandes par le réseau (au format JSON) et "
    "renvoie une réponse, sans interface. Chaque **endpoint** est un guichet précis : `GET /sante`, "
    "`GET /modele`, `POST /predire`, `POST /expliquer`, `GET /explications/globales`. L'application l'utilise "
    "pour l'explication d'un client (page 5) ; d'autres logiciels pourraient l'utiliser de la même façon."
)

url = api_client.api_url()
try:
    health = api_client.sante()
except api_client.ApiIndisponible as exc:
    st.error(f"L'API ne répond pas à l'adresse {url} : {exc}.")
    st.markdown("Pour la lancer, dans un autre terminal, depuis la racine du projet :")
    st.code(COMMANDE, language="powershell")
    st.stop()

st.success(f"L'API répond à l'adresse **{url}** · modèles : {', '.join(health['modeles_disponibles'])}")
st.markdown(f"Documentation interactive (essayer chaque endpoint depuis le navigateur) : [{url}/docs]({url}/docs)")
st.caption("Versions côté API : " + " · ".join(f"{k} {v}" for k, v in health["versions"].items()))

modele = st.radio("Modèle demandé à l'API", list(MODELES), format_func=MODELES.get, horizontal=True)
info = api_client.informations(modele)
c1, c2, c3 = st.columns(3)
c1.metric("Seuil recommandé", f"{info['seuil_recommande']:.4f}")
c2.metric("AUC validation croisée", f"{info['auc_validation_croisee']:.4f}")
c3.metric(f"AUC test ({info['clients_test']} clients)", f"{info['auc_test']:.4f}")
st.caption(f"Seuil : {info['methode_seuil']}.")
if info["variables_ignorees"]:
    st.info(f"Ce modèle reçoit les 23 variables mais ignore : {', '.join(info['variables_ignorees'])}.")

results_path = config.MODELS_DIR / "resultats.json"
X_test = test_clients(str(results_path), results_path.stat().st_mtime)
client_id = int(st.number_input("Identifiant d'un client du jeu de test", min_value=int(X_test.index.min()),
                                max_value=int(X_test.index.max()), value=1358, step=1))
if client_id not in X_test.index:
    st.warning("Cet identifiant n'est pas dans le jeu de test.")
    st.stop()
client = {k: int(v) for k, v in X_test.loc[client_id].items()}

col_req, col_rep = st.columns(2)
endpoint = col_req.radio("Endpoint", ["POST /predire", "POST /expliquer"], horizontal=True)
body = {"client": client, "modele": modele}
if endpoint == "POST /expliquer":
    body.update({"methodes": ["shap", "lime"], "nombre_variables": 5})
col_req.markdown("**Requête envoyée (JSON)**")
col_req.json(body)
if col_req.button("Envoyer la requête", type="primary"):
    try:
        if endpoint == "POST /predire":
            response, ms = api_client.predire(client, modele)
        else:
            response, ms = api_client.expliquer(client, modele)
        col_rep.markdown(f"**Réponse de l'API** (temps total vu par l'application : {ms:.0f} ms, "
                         f"calcul côté API : {response['duree_ms']:.0f} ms)")
        col_rep.json(response)
    except api_client.ApiIndisponible as exc:
        col_rep.error(f"Échec de la requête : {exc}")
