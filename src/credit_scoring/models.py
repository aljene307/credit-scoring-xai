"""Les quatre modèles, leurs Pipelines et leurs grilles d'hyperparamètres (choix M01, M03, M06)."""

from math import prod

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


def default_grid(name: str) -> dict:
    """Grille de référence, sans le préfixe du Pipeline (choix M06).

    Pas de pondération des classes (choix M03) : elle n'améliorait pas l'AUC des
    meilleures combinaisons et faussait les probabilités. Les grilles de l'arbre
    et de XGBoost ont été prolongées pour que les meilleures valeurs ne soient
    plus en bord de grille.
    """
    grids = {
        "logreg": {"C": [0.01, 0.1, 1, 10]},
        "arbre": {
            "max_depth": [3, 5, 8, 12, 16, 20],
            "min_samples_leaf": [1, 20, 50, 100, 200, 400, 800, 1600],
        },
        "random_forest": {
            "max_depth": [8, 12, None],
            "min_samples_leaf": [1, 10],
            "max_features": ["sqrt", 0.5],
        },
        # Centrée sur l'optimum d'une première grille prolongée (500 arbres, taux 0,02,
        # profondeur 4, sous-échantillon 0,6) : une valeur de chaque côté.
        "xgboost": {
            "n_estimators": [300, 500, 800],
            "learning_rate": [0.01, 0.02, 0.03],
            "max_depth": [3, 4, 5],
            "subsample": [0.4, 0.6, 0.8],
        },
    }
    if name not in grids:
        raise ValueError(f"Modèle inconnu : {name}")
    return grids[name]


def grid_edges(grid: dict, best_params: dict) -> list[str]:
    """Hyperparamètres numériques dont la meilleure valeur est la plus petite ou la
    plus grande de la grille : le vrai optimum est peut-être au-delà.

    None (profondeur sans limite) compte comme la plus grande valeur.
    """
    edges = []
    for key, values in grid.items():
        numeric = [v for v in values if v is not None]
        if len(values) < 2 or not all(isinstance(v, (int, float)) for v in numeric):
            continue
        ordered = sorted(numeric) + ([None] if None in values else [])
        if best_params[key] in (ordered[0], ordered[-1]):
            edges.append(key)
    return edges


def to_pipeline_grid(grid: dict) -> dict:
    """Ajoute le préfixe « modele__ » attendu par GridSearchCV sur un Pipeline."""
    return {f"{STEP}__{key}": values for key, values in grid.items()}


def n_combinations(grid: dict) -> int:
    return prod(len(values) for values in grid.values())
