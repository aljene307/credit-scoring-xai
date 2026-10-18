"""Le « moteur » de l'API : charge une fois les deux modèles XGBoost et leurs explicateurs, puis
prédit et explique un client. Les mêmes fonctions que l'application (credit_scoring) sont
utilisées, avec les mêmes réglages : les chiffres sont identiques à ceux de la page 5.

Au démarrage, l'API ne lit que les deux modèles XGBoost et le paquet de démarrage préparé par
scripts/expliquer.py (choix API05) : ni téléchargement, ni jeu de données.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import shap
import sklearn
import xgboost

from credit_scoring.config import CATEGORICAL_FEATURES
from credit_scoring.explain import (LIME_FEATURES, LIME_SAMPLES, LIME_SEED, LimeExplainer, ShapExplainer,
                                    agreement, summary_sentence, top_features)
from credit_scoring.global_explanations import API_BUNDLE, load_global
from credit_scoring.training import load_model

FICHIERS = {"complet": "xgboost.joblib", "sans_sensibles": "xgboost_sans_sensibles.joblib"}

NOMS = {"complet": "XGBoost (23 variables)", "sans_sensibles": "XGBoost sans variables sensibles (19 variables)"}


@dataclass
class Variante:
    """Un modèle servi par l'API, avec tout ce qu'il faut pour le prédire et l'expliquer."""

    pipeline: object
    variables: list[str]
    seuil: float
    auc_validation: float
    auc_test: float
    shap: ShapExplainer
    lime: LimeExplainer
    base_probabilite: float


class Service:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        bundle = load_model(self.models_dir / API_BUNDLE)
        self.toutes_variables = bundle["variables"]
        self.median_limit = bundle["mediane_plafond"]
        self.clients_test = bundle["clients_test"]
        self.variantes = {}
        for name, fichier in FICHIERS.items():
            pipeline, v = load_model(self.models_dir / fichier), bundle["variantes"][name]
            categorical = [c for c in CATEGORICAL_FEATURES if c in v["variables"]]
            self.variantes[name] = Variante(
                pipeline=pipeline,
                variables=v["variables"],
                seuil=v["seuil"],
                auc_validation=v["auc_validation"],
                auc_test=v["auc_test"],
                # Mêmes 500 clients de référence que l'application (X03) et statistiques de LIME (API05).
                shap=ShapExplainer(pipeline, v["fond"], categorical),
                lime=LimeExplainer.from_stats(v["lime"], v["variables"], categorical),
                base_probabilite=float(pipeline.predict_proba(v["fond"])[:, 1].mean()),
            )
        self.versions = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                         "shap": shap.__version__}

    def ignorees(self, modele: str) -> list[str]:
        return [c for c in self.toutes_variables if c not in self.variantes[modele].variables]

    def predire(self, client: dict, modele: str, seuil: float | None = None) -> dict:
        start = time.perf_counter()
        v = self.variantes[modele]
        row = pd.DataFrame([client])[v.variables]
        probabilite = float(v.pipeline.predict_proba(row)[:, 1][0])
        seuil = v.seuil if seuil is None else seuil
        return {
            "modele": modele,
            "probabilite": probabilite,
            "seuil": seuil,
            "signale": probabilite >= seuil,
            "decision": "signalé comme risqué" if probabilite >= seuil else "non signalé",
            "variables_ignorees": self.ignorees(modele),
            "duree_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    def expliquer(self, client: dict, modele: str, methodes: list[str], k: int) -> dict:
        start = time.perf_counter()
        v = self.variantes[modele]
        result = self.predire(client, modele)
        row = pd.DataFrame([client])[v.variables]
        serie = row.iloc[0]
        contributions = None
        if "shap" in methodes:
            values, base = v.shap.explain(row)
            contributions = values.iloc[0]
            result.update({
                "valeur_de_base": base,
                "contributions_shap": {f: float(contributions[f]) for f in top_features(contributions, len(contributions))},
                "principales_shap": top_features(contributions, k),
                "phrase": summary_sentence(contributions, serie, result["probabilite"], v.base_probabilite,
                                           v.seuil, self.median_limit),
            })
        if "lime" in methodes:
            # LIME ajuste un modèle différent selon le nombre de variables : on garde les 10 de
            # l'application (au moins), puis on ne renvoie que les k premières règles.
            weights, rules = v.lime.explain(v.pipeline, serie, num_samples=LIME_SAMPLES, seed=LIME_SEED,
                                            num_features=max(LIME_FEATURES, k))
            result.update({
                "regles_lime": [{"regle": r, "poids": float(w)} for r, w in list(rules.items())[:k]],
                "poids_lime": {f: float(w) for f, w in weights.items()},
            })
            if contributions is not None:
                result["accord"] = agreement(contributions, weights, k=min(5, k))
        result["duree_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return result

    def informations(self, modele: str, descriptions: dict[str, str]) -> dict:
        v = self.variantes[modele]
        return {
            "modele": modele,
            "nom": NOMS[modele],
            "variables": {f: {"description": descriptions.get(f, ""), "utilisee": f in v.variables}
                          for f in self.toutes_variables},
            "variables_ignorees": self.ignorees(modele),
            "seuil_recommande": v.seuil,
            "methode_seuil": "F1 maximal sur les prédictions hors pli de l'entraînement (validation croisée à 5 plis)",
            "auc_validation_croisee": v.auc_validation,
            "auc_test": v.auc_test,
            "clients_test": self.clients_test,
        }

    def globales(self) -> dict:
        g = load_global(self.models_dir)
        values = g["valeurs_shap"]["xgboost"]
        permutation = {m: dict(zip(t["variable"], t["moyenne"].astype(float)))
                       for m, t in g["permutation"].groupby("modele")}
        mean_abs = values.abs().mean().sort_values(ascending=False)
        return {
            "modele": "complet",
            "echelle": g["shap"]["xgboost"]["echelle"],
            "valeur_de_base": g["shap"]["xgboost"]["valeur_de_base"],
            "clients_expliques": int(len(values)),
            "contribution_moyenne": {f: float(v) for f, v in mean_abs.items()},
            "permutation": permutation,
        }
