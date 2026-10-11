"""Seuils de décision et métriques (choix E01, E02)."""

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_curve


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


def threshold_report(y_true, proba) -> dict:
    """Seuil recommandé (F1 maximal) et seuil du KS, avec leurs métriques (choix E01)."""
    ks_t, ks_value = ks_threshold(y_true, proba)
    return {
        "f1_max": metrics_at_threshold(y_true, proba, best_f1_threshold(y_true, proba)),
        "ks": {**metrics_at_threshold(y_true, proba, ks_t), "ks": ks_value},
    }
