"""API d'interprétation du scoring de crédit (module 6).

Lancement, depuis la racine du projet :
    .venv\\Scripts\\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
Documentation interactive : http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from api.schemas import (
    Client,
    DemandeExplication,
    DemandePrediction,
    Explication,
    ExplicationsGlobales,
    InformationsModele,
    ModeleNom,
    Prediction,
    Sante,
)
from api.service import Service
from credit_scoring import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Au démarrage : chargement unique des modèles et des explicateurs (quelques secondes)."""
    try:
        app.state.service = Service(config.MODELS_DIR)
        app.state.erreur = None
    except FileNotFoundError as exc:
        app.state.service = None
        app.state.erreur = (f"Modèles introuvables ({exc.filename}) : lancez scripts/train.py puis "
                            "scripts/expliquer.py.")
    yield


app = FastAPI(
    title="API de scoring de crédit explicable",
    description=(
        "Prédit le défaut de paiement d'un client de carte de crédit (jeu Default of Credit Card Clients, "
        "UCI 350) et explique la décision avec SHAP et LIME. Deux modèles XGBoost : `complet` (23 variables) "
        "et `sans_sensibles` (sans SEX, AGE, MARRIAGE, EDUCATION), recommandé au déploiement."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def service(request: Request) -> Service:
    if request.app.state.service is None:
        raise HTTPException(status_code=503, detail=request.app.state.erreur)
    return request.app.state.service


def descriptions() -> dict[str, str]:
    return {name: field.description or "" for name, field in Client.model_fields.items()}


@app.get("/sante", response_model=Sante, summary="L'API répond-elle ?")
def sante(request: Request):
    s = service(request)
    return {"statut": "ok", "modeles_disponibles": list(s.variantes), "versions": s.versions}


@app.get("/modele", response_model=InformationsModele, summary="Caractéristiques d'un modèle")
def modele(request: Request, modele: ModeleNom = "sans_sensibles"):
    """Variables attendues (et celles que le modèle ignore), seuil recommandé, AUC."""
    return service(request).informations(modele, descriptions())


@app.post("/predire", response_model=Prediction, summary="Probabilité de défaut d'un client")
def predire(request: Request, demande: DemandePrediction):
    """Les 23 variables sont toujours attendues ; le modèle `sans_sensibles` ignore les 4 variables
    sensibles et les liste dans `variables_ignorees`."""
    return service(request).predire(demande.client.model_dump(), demande.modele, demande.seuil)


@app.post("/expliquer", response_model=Explication, summary="Pourquoi ce score ?")
def expliquer(request: Request, demande: DemandeExplication):
    """Contributions SHAP (en probabilité : valeur de base + somme = probabilité), règles LIME,
    accord entre les deux méthodes et phrase de synthèse."""
    return service(request).expliquer(demande.client.model_dump(), demande.modele, demande.methodes,
                                      demande.nombre_variables)


@app.get("/explications/globales", response_model=ExplicationsGlobales, summary="Explication globale du modèle")
def explications_globales(request: Request):
    """Contribution SHAP moyenne de chaque variable (1 000 clients du test, modèle complet) et importance
    par permutation des quatre modèles."""
    return service(request).globales()
