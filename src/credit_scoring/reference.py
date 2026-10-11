"""Modèles de référence (entraînés par scripts/train.py) et jeu de test correspondant."""

from pathlib import Path

import pandas as pd

from credit_scoring import config
from credit_scoring.data import load_credit_data, recode_undocumented
from credit_scoring.preprocessing import split_data
from credit_scoring.training import load_model, load_results


def load_reference(models_dir: Path | None = None) -> tuple[dict, dict]:
    """Résultats (resultats.json) et modèles chargés, par nom de modèle."""
    models_dir = Path(models_dir or config.MODELS_DIR)
    results = load_results(models_dir / "resultats.json")
    models = {name: load_model(models_dir / f"{name}.joblib") for name in results["modeles"]}
    return results, models


def reference_split(meta: dict, df: pd.DataFrame | None = None):
    """Reconstitue la séparation entraînement / test utilisée à l'entraînement.

    Mêmes données recodées, mêmes variables, même part de test, même graine :
    le jeu de test est exactement celui qui a été mis de côté. Vérifie le nombre
    de lignes d'entraînement enregistré pour détecter toute divergence.
    """
    df = df if df is not None else recode_undocumented(load_credit_data())
    X_train, X_test, y_train, y_test = split_data(
        df, meta["cible"], meta["variables"], meta["part_test"], meta["graine"]
    )
    if len(X_train) != meta["lignes_entrainement"]:
        raise ValueError(
            f"Séparation différente de celle de l'entraînement : {len(X_train)} lignes "
            f"au lieu de {meta['lignes_entrainement']}."
        )
    return X_train, X_test, y_train, y_test
