"""Les quatre modèles, leurs Pipelines et leurs grilles d'hyperparamètres (choix M01, M03, M06)."""

from math import prod

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from credit_scoring.config import RANDOM_STATE
from credit_scoring.preprocessing import build_preprocessor

MODEL_LABELS = {
    "logreg": "Régression logistique",
    "arbre": "Arbre de décision",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
# Seule la régression logistique reçoit des variables standardisées (choix P05).
SCALED_MODELS = {"logreg"}
STEP = "modele"  # nom de l'étape du modèle dans le Pipeline


def make_estimator(name: str):
    """Modèle non réglé. Random Forest et XGBoost utilisent un seul cœur chacun :
    c'est GridSearchCV qui répartit les entraînements sur les cœurs."""
    if name == "logreg":
        return LogisticRegression(max_iter=2000)
    if name == "arbre":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=RANDOM_STATE)
    if name == "xgboost":
        return XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=RANDOM_STATE)
    raise ValueError(f"Modèle inconnu : {name}")


def build_pipeline(name: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    """Pipeline prétraitement -> modèle, réappris en entier sur chaque pli (choix P03)."""
    preprocessor = build_preprocessor(numeric, categorical, scale=name in SCALED_MODELS)
    return Pipeline([("pretraitement", preprocessor), (STEP, make_estimator(name))])


def class_ratio(y_train: pd.Series) -> float:
    """Nombre de bons payeurs par défaut : valeur « pondérée » de scale_pos_weight."""
    return round(float((y_train == 0).sum() / (y_train == 1).sum()), 2)


def default_grid(name: str, y_train: pd.Series) -> dict:
    """Grille de référence, sans le préfixe du Pipeline (choix M06)."""
    grids = {
        "logreg": {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]},
        "arbre": {
            "max_depth": [3, 5, 8, 12],
            "min_samples_leaf": [1, 20, 100],
            "class_weight": [None, "balanced"],
        },
        "random_forest": {
            "max_depth": [8, 12, None],
            "min_samples_leaf": [1, 10],
            "max_features": ["sqrt", 0.5],
            "class_weight": [None, "balanced_subsample"],
        },
        "xgboost": {
            "n_estimators": [200, 400],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0],
            "scale_pos_weight": [1, class_ratio(y_train)],
        },
    }
    if name not in grids:
        raise ValueError(f"Modèle inconnu : {name}")
    return grids[name]


def to_pipeline_grid(grid: dict) -> dict:
    """Ajoute le préfixe « modele__ » attendu par GridSearchCV sur un Pipeline."""
    return {f"{STEP}__{key}": values for key, values in grid.items()}


def n_combinations(grid: dict) -> int:
    return prod(len(values) for values in grid.values())
