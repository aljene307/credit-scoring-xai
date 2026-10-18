"""Explications globales calculées une fois hors ligne (choix X04) : SHAP sur 1 000 clients du test,
importance par permutation des quatre modèles, analyse d'équité. Lu par l'application."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict, cross_val_score

from credit_scoring.config import RANDOM_STATE, SENSITIVE_FEATURES
from credit_scoring.evaluation import best_f1_threshold
from credit_scoring.explain import ShapExplainer, background_sample
from credit_scoring.fairness import group_errors, proxy_scores, sensitive_groups
from credit_scoring.models import build_pipeline
from credit_scoring.training import make_cv, save_model

SHAP_MODELS = ("xgboost", "logreg")  # choix X05
SUMMARY = "explications.json"


def reference_background(pipeline, X_train: pd.DataFrame, size: int = 500) -> pd.DataFrame:
    """Échantillon de référence de SHAP : stratifié sur le score du modèle de référence (choix X03)."""
    return background_sample(X_train, size, RANDOM_STATE, scores=pipeline.predict_proba(X_train)[:, 1])


def compute_and_save(
    models_dir: Path,
    models: dict,
    results: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical: list[str],
    n_clients: int = 1000,
    background_size: int = 500,
    n_repeats: int = 10,
    log=print,
) -> dict:
    models_dir = Path(models_dir)
    summary = {"durees_s": {}, "versions": {"python": platform.python_version(), "shap": shap.__version__,
                                            "scikit-learn": sklearn.__version__}}
    reference = models["xgboost"]
    background = reference_background(reference, X_train, background_size)
    rng = np.random.default_rng(RANDOM_STATE)
    sample = X_test.iloc[np.sort(rng.choice(len(X_test), min(n_clients, len(X_test)), replace=False))]
    summary.update({"clients_expliques": len(sample), "clients_reference": len(background),
                    "tirage_reference": "stratifié sur le score de XGBoost", "graine": RANDOM_STATE})

    # SHAP global
    summary["shap"] = {}
    for name in SHAP_MODELS:
        start = time.perf_counter()
        explainer = ShapExplainer(models[name], background, categorical)
        values, base = explainer.explain(sample)
        values.to_parquet(models_dir / f"shap_{name}.parquet")
        summary["shap"][name] = {"echelle": explainer.scale, "valeur_de_base": base}
        summary["durees_s"][f"shap_{name}"] = round(time.perf_counter() - start, 1)
        log(f"SHAP {name} : {len(sample)} clients en {summary['durees_s'][f'shap_{name}']} s")

    # Importance par permutation sur le test (choix X06)
    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        result = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
                                        random_state=RANDOM_STATE, n_jobs=-1)
        rows += [{"modele": name, "variable": var, "moyenne": m, "ecart_type": s}
                 for var, m, s in zip(X_test.columns, result.importances_mean, result.importances_std)]
        summary["durees_s"][f"permutation_{name}"] = round(time.perf_counter() - start, 1)
        log(f"permutation {name} : {summary['durees_s'][f'permutation_{name}']} s")
    pd.DataFrame(rows).to_parquet(models_dir / "permutation.parquet")

    # Équité (choix X09)
    start = time.perf_counter()
    sensitive = [c for c in SENSITIVE_FEATURES if c in X_train]
    others = [c for c in X_train.columns if c not in sensitive]
    cv = make_cv()
    without = build_pipeline("xgboost", [c for c in others if c not in categorical],
                             [c for c in others if c in categorical])
    best = results["modeles"]["xgboost"]["meilleurs_hyperparametres"]
    without.set_params(**{f"modele__{k}": v for k, v in best.items()})
    auc_without = cross_val_score(without, X_train[others], y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    oof = cross_val_predict(clone(without), X_train[others], y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold_without = best_f1_threshold(y_train, oof)
    without.fit(X_train[others], y_train)
    save_model(without, models_dir / "xgboost_sans_sensibles.joblib")

    threshold = results["modeles"]["xgboost"]["seuils"]["f1_max"]["seuil"]
    variants = {
        "avec": (reference.predict_proba(X_test)[:, 1], threshold),
        "sans": (without.predict_proba(X_test[others])[:, 1], threshold_without),
    }
    fairness = {
        "variables_sensibles": sensitive,
        "auc_validation_sans": {"moyenne": float(auc_without.mean()), "ecart_type": float(auc_without.std())},
        "auc_validation_avec": results["modeles"]["xgboost"]["validation_croisee"]["roc_auc"],
        "seuil_sans": threshold_without,
        "relais": proxy_scores(X_train, sensitive, cv).to_dict("records"),
        "groupes": {},
    }
    for gname, groups in sensitive_groups(X_test).items():
        fairness["groupes"][gname] = {}
        for variant, (proba, t) in variants.items():
            table, tests = group_errors(y_test, proba, t, groups)
            fairness["groupes"][gname][variant] = {"tableau": table.to_dict("records"), "tests": tests}
    summary["equite"] = fairness
    summary["durees_s"]["equite"] = round(time.perf_counter() - start, 1)
    log(f"équité : {summary['durees_s']['equite']} s")

    (models_dir / SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_global(models_dir: Path) -> dict | None:
    """Résumé et valeurs SHAP globales, ou None si le script n'a pas encore été lancé."""
    models_dir = Path(models_dir)
    path = models_dir / SUMMARY
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["valeurs_shap"] = {name: pd.read_parquet(models_dir / f"shap_{name}.parquet") for name in summary["shap"]}
    summary["permutation"] = pd.read_parquet(models_dir / "permutation.parquet")
    return summary
