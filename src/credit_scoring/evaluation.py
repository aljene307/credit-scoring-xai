"""Seuils de décision et métriques (choix E01 à E06)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from credit_scoring.config import RANDOM_STATE


def best_f1_threshold(y_true, proba) -> float:
    """Seuil qui maximise le F1 de la classe positive."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # le dernier point n'a pas de seuil
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    return float(thresholds[np.argmax(f1)])


def ks_threshold(y_true, proba) -> tuple[float, float]:
    """Seuil du KS et valeur du KS : écart maximal entre les taux de vrais et de faux positifs.

    Le KS est la plus grande distance entre les distributions de scores des
    défauts et des bons payeurs.
    """
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    gap = tpr - fpr
    best = int(np.argmax(gap))
    return float(thresholds[best]), float(gap[best])


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    """Précision, rappel et F1 de la classe positive, et part des clients signalés."""
    predicted = (np.asarray(proba) >= threshold).astype(int)
    return {
        "seuil": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "rappel": float(recall_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted)),
        "part_signales": float(predicted.mean()),
    }


def confusion_counts(y_true, proba, threshold: float) -> dict:
    """Matrice de confusion : vrais/faux positifs et négatifs (positif = défaut prédit)."""
    predicted = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)}


def holdout_report(y_true, proba, threshold: float) -> dict:
    """Métriques sur le jeu de test (choix E02) au seuil fixé à l'avance."""
    auc = float(roc_auc_score(y_true, proba))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_threshold(y_true, proba)[1],
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "proba_moyenne": float(np.mean(proba)),
        **metrics_at_threshold(y_true, proba, threshold),
        "confusion": confusion_counts(y_true, proba, threshold),
    }


def bootstrap_auc(y_true, probas: dict, n_resamples: int = 1000, seed: int = RANDOM_STATE) -> dict:
    """AUC de chaque modèle sur n tirages avec remise des clients du test (choix E05).

    Les mêmes tirages servent à tous les modèles : les écarts entre modèles sont
    donc comparés sur exactement les mêmes clients (bootstrap apparié).
    """
    y = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in probas}
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():  # tirage sans l'une des classes : AUC non définie
            continue
        for name, proba in probas.items():
            samples[name].append(roc_auc_score(y[idx], np.asarray(proba)[idx]))
    return {name: np.array(values) for name, values in samples.items()}


def confidence_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Intervalle par percentiles des valeurs bootstrap."""
    alpha = (1 - level) / 2 * 100
    return float(np.percentile(values, alpha)), float(np.percentile(values, 100 - alpha))


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Probabilité moyenne prédite et taux de défaut observé par tranche de même effectif (E03)."""
    frame = pd.DataFrame({"proba": np.asarray(proba), "defaut": np.asarray(y_true)})
    frame["tranche"] = pd.qcut(frame["proba"], q=n_bins, labels=False, duplicates="drop")
    table = frame.groupby("tranche").agg(
        clients=("defaut", "size"), proba_moyenne=("proba", "mean"), taux_observe=("defaut", "mean")
    )
    return table.reset_index(drop=True)


def decision_sentence(counts: dict) -> str:
    """Phrase en langage de décideur, à partir de la matrice de confusion (choix E06)."""
    def n(value: int) -> str:  # 6000 -> « 6 000 »
        return f"{value:,}".replace(",", " ")

    flagged = counts["vp"] + counts["fp"]
    defaults = counts["vp"] + counts["fn"]
    total = flagged + counts["vn"] + counts["fn"]
    share = counts["vp"] / flagged if flagged else 0
    return (
        f"Sur {n(total)} clients, le modèle en signale {n(flagged)} comme risqués : "
        f"{n(counts['vp'])} d'entre eux font réellement défaut ({share * 100:.0f} %). "
        f"Il manque {n(counts['fn'])} défauts sur {n(defaults)}."
    )


def threshold_report(y_true, proba) -> dict:
    """Seuil recommandé (F1 maximal) et seuil du KS, avec leurs métriques (choix E01)."""
    ks_t, ks_value = ks_threshold(y_true, proba)
    return {
        "f1_max": metrics_at_threshold(y_true, proba, best_f1_threshold(y_true, proba)),
        "ks": {**metrics_at_threshold(y_true, proba, ks_t), "ks": ks_value},
    }
