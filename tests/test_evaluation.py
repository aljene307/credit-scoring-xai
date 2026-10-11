"""Tests des métriques de l'étape 4 et de la reconstitution du jeu de test."""

import numpy as np
import pandas as pd
import pytest

from credit_scoring import evaluation
from credit_scoring.reference import reference_split

Y = np.array([0, 0, 0, 0, 1, 1])
PROBA = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])


def test_confusion_counts_on_a_hand_checkable_case():
    # Seuil 0.5 : signalés = 0.6 (bon payeur), 0.7 et 0.9 (défauts).
    assert evaluation.confusion_counts(Y, PROBA, 0.5) == {"vn": 3, "fp": 1, "fn": 0, "vp": 2}


def test_holdout_report_contains_all_metrics():
    report = evaluation.holdout_report(Y, PROBA, 0.5)
    assert report["auc"] == pytest.approx(1.0)
    assert report["gini"] == pytest.approx(2 * report["auc"] - 1)
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["rappel"] == 1.0
    assert report["brier"] == pytest.approx(np.mean((PROBA - Y) ** 2))
    assert report["confusion"]["fp"] == 1


def test_decision_sentence_gives_counts_in_plain_words():
    sentence = evaluation.decision_sentence({"vn": 4000, "fp": 650, "fn": 570, "vp": 780})
    assert sentence == (
        "Sur 6 000 clients, le modèle en signale 1 430 comme risqués : 780 d'entre eux font "
        "réellement défaut (55 %). Il manque 570 défauts sur 1 350."
    )


def test_bootstrap_is_paired_and_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    good = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    samples = evaluation.bootstrap_auc(y, {"a": good, "b": good}, n_resamples=200)
    assert len(samples["a"]) == len(samples["b"]) == 200
    np.testing.assert_allclose(samples["a"] - samples["b"], 0)  # mêmes tirages, mêmes clients
    low, high = evaluation.confidence_interval(samples["a"])
    from sklearn.metrics import roc_auc_score
    assert low < roc_auc_score(y, good) < high


def test_calibration_table_uses_equal_size_bins_and_matches_a_calibrated_model():
    rng = np.random.default_rng(2)
    proba = rng.random(20_000)
    y = (rng.random(20_000) < proba).astype(int)  # parfaitement calibré par construction
    table = evaluation.calibration_table(y, proba, n_bins=10)
    assert len(table) == 10
    assert table["clients"].tolist() == [2000] * 10
    np.testing.assert_allclose(table["taux_observe"], table["proba_moyenne"], atol=0.03)


def test_reference_split_rebuilds_the_training_split_and_detects_divergence():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.random(100), "DEFAULT": rng.integers(0, 2, 100)})
    meta = {"cible": "DEFAULT", "variables": ["x"], "part_test": 0.2, "graine": 42, "lignes_entrainement": 80}
    X_train, X_test, _, y_test = reference_split(meta, df)
    assert len(X_test) == 20 and len(y_test) == 20
    with pytest.raises(ValueError, match="Séparation différente"):
        reference_split({**meta, "lignes_entrainement": 75}, df)
