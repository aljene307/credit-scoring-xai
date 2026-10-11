"""Optimisation des hyperparamètres par GridSearchCV et sauvegarde des modèles (étape 3)."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from credit_scoring.config import RANDOM_STATE
from credit_scoring.evaluation import threshold_report
from credit_scoring.models import (
    MODEL_LABELS,
    STEP,
    build_pipeline,
    default_grid,
    n_combinations,
    to_pipeline_grid,
)

CV_FOLDS = 5
# Métrique de choix : AUC (choix M02). Les autres sont enregistrées pour le tableau comparatif.
REFIT_METRIC = "roc_auc"
SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "rappel": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
}


def make_cv() -> StratifiedKFold:
    """5 plis stratifiés, mélangés avec une graine fixe : mêmes plis pour tous les modèles."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _plain(value):
    """Valeur sérialisable en JSON (types NumPy -> types Python)."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def train_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric: list[str],
    categorical: list[str],
    grid: dict | None = None,
    n_jobs: int = -1,
) -> tuple[GridSearchCV, dict]:
    """GridSearchCV sur le jeu d'entraînement seulement, puis seuils sur les prédictions hors pli.

    Retourne la recherche ajustée (meilleur modèle réentraîné sur tout
    l'entraînement) et un résumé sérialisable en JSON.
    """
    grid = grid if grid is not None else default_grid(name, y_train)
    search = GridSearchCV(
        build_pipeline(name, numeric, categorical),
        to_pipeline_grid(grid),
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=make_cv(),
        n_jobs=n_jobs,
        return_train_score=True,
    )
    start = time.perf_counter()
    search.fit(X_train, y_train)
    search_seconds = time.perf_counter() - start

    # Prédictions hors pli : chaque client est prédit par un modèle qui ne l'a pas vu.
    # Le seuil est choisi sur ces prédictions, jamais sur le test (choix E01).
    oof_proba = cross_val_predict(
        clone(search.best_estimator_), X_train, y_train, cv=make_cv(), method="predict_proba", n_jobs=n_jobs
    )[:, 1]

    best = search.best_index_
    results = search.cv_results_
    summary = {
        "modele": name,
        "libelle": MODEL_LABELS[name],
        "combinaisons": n_combinations(grid),
        "entrainements": n_combinations(grid) * CV_FOLDS,
        "duree_s": round(search_seconds, 1),
        "meilleurs_hyperparametres": {
            key.removeprefix(f"{STEP}__"): _plain(value) for key, value in search.best_params_.items()
        },
        "validation_croisee": {
            metric: {
                "moyenne": float(results[f"mean_test_{metric}"][best]),
                "ecart_type": float(results[f"std_test_{metric}"][best]),
            }
            for metric in SCORING
        },
        "auc_entrainement": float(results["mean_train_roc_auc"][best]),
        "seuils": threshold_report(y_train, oof_proba),
    }
    return search, summary


def cv_results_table(search: GridSearchCV) -> pd.DataFrame:
    """Toutes les combinaisons essayées, de la meilleure à la moins bonne (AUC de validation)."""
    results = pd.DataFrame(search.cv_results_)
    params = results["params"].apply(
        lambda p: {k.removeprefix(f"{STEP}__"): v for k, v in p.items()}
    ).apply(pd.Series)
    # Une colonne qui mélange texte et nombres (ex. max_features : "sqrt" et 0.5)
    # ne peut pas être enregistrée en Parquet : ses valeurs sont converties en texte.
    # Les valeurs absentes (None, devenu NaN) restent vides et ne comptent pas comme un type.
    for col in params.columns:
        if len({type(v) for v in params[col] if not pd.isna(v)}) > 1:
            params[col] = params[col].map(lambda v: None if pd.isna(v) else str(v))
    scores = results[[f"mean_test_{m}" for m in SCORING] + ["mean_train_roc_auc", "mean_fit_time"]]
    table = pd.concat([params, scores], axis=1).sort_values("mean_test_roc_auc", ascending=False)
    return table.reset_index(drop=True)


def save_model(model, path: Path) -> int:
    """Sauvegarde compressée (joblib, niveau 3) ; retourne la taille en octets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path.stat().st_size


def load_model(path: Path):
    return joblib.load(path)


def save_results(results: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def load_results(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
