"""Tests des modèles, des grilles, des seuils et de l'entraînement (données synthétiques)."""

import json

import numpy as np
import pandas as pd
import pytest

from credit_scoring import evaluation, models, training


def make_data(n: int = 400):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10_000, 500_000, n).astype(float),
            "PAY_0": rng.choice([-2, -1, 0, 1, 2], n).astype(float),
            "SEX": rng.choice([1, 2], n),
        }
    )
    risk = 0.15 + 0.5 * (X["PAY_0"] >= 1)
    y = pd.Series((rng.random(n) < risk).astype(int), name="DEFAULT")
    return X, y


@pytest.mark.parametrize(
    "name, expected", [("logreg", 8), ("arbre", 24), ("random_forest", 24), ("xgboost", 32)]
)
def test_reference_grids_have_the_announced_number_of_combinations(name, expected):
    _, y = make_data()
    assert models.n_combinations(models.default_grid(name, y)) == expected


def test_scale_pos_weight_uses_the_class_ratio():
    y = pd.Series([0] * 78 + [1] * 22)
    assert models.class_ratio(y) == 3.55
    assert models.default_grid("xgboost", y)["scale_pos_weight"] == [1, 3.55]


def test_only_logistic_regression_is_standardised():
    X, _ = make_data()
    for name in models.MODEL_LABELS:
        pipe = models.build_pipeline(name, ["LIMIT_BAL", "PAY_0"], ["SEX"])
        steps = pipe.named_steps["pretraitement"].transformers[0][1].named_steps
        assert ("standardisation" in steps) == (name == "logreg")


def test_thresholds_on_a_hand_checkable_case():
    y = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    # Au seuil 0.7 : 2 signalés, tous deux défauts -> précision 1, rappel 1, F1 1.
    assert evaluation.best_f1_threshold(y, proba) == pytest.approx(0.7)
    threshold, ks = evaluation.ks_threshold(y, proba)
    assert threshold == pytest.approx(0.7) and ks == pytest.approx(1.0)
    m = evaluation.metrics_at_threshold(y, proba, 0.6)
    assert m["precision"] == pytest.approx(2 / 3) and m["rappel"] == 1.0
    assert m["part_signales"] == pytest.approx(0.5)


def test_train_model_on_small_grids_and_save_reload(tmp_path):
    X, y = make_data()
    small_grids = {
        "logreg": {"C": [0.1, 1]},
        "arbre": {"max_depth": [2, 3]},
        "random_forest": {"max_depth": [3], "n_estimators": [20]},
        "xgboost": {"n_estimators": [20], "max_depth": [2]},
    }
    for name, grid in small_grids.items():
        search, summary = training.train_model(name, X, y, ["LIMIT_BAL", "PAY_0"], ["SEX"], grid=grid, n_jobs=1)
        assert summary["combinaisons"] == models.n_combinations(grid)
        assert set(summary["meilleurs_hyperparametres"]) == set(grid)
        assert 0.5 < summary["validation_croisee"]["roc_auc"]["moyenne"] <= 1
        assert set(summary["seuils"]) == {"f1_max", "ks"}
        json.dumps(summary)  # résumé sérialisable

        size = training.save_model(search.best_estimator_, tmp_path / f"{name}.joblib")
        assert size > 0
        reloaded = training.load_model(tmp_path / f"{name}.joblib")
        np.testing.assert_allclose(reloaded.predict_proba(X), search.best_estimator_.predict_proba(X))

        table = training.cv_results_table(search)
        assert len(table) == models.n_combinations(grid)
        assert table["mean_test_roc_auc"].is_monotonic_decreasing
        table.to_parquet(tmp_path / f"grille_{name}.parquet")


def test_cv_table_with_mixed_types_can_be_saved_as_parquet(tmp_path):
    """Cas réel : max_features mélange "sqrt" (texte) et 0.5 (nombre), avec None ailleurs."""
    X, y = make_data()
    grid = {"max_depth": [3, None], "max_features": ["sqrt", 0.5], "class_weight": [None, "balanced_subsample"],
            "n_estimators": [10]}
    search, _ = training.train_model("random_forest", X, y, ["LIMIT_BAL", "PAY_0"], ["SEX"], grid=grid, n_jobs=1)
    table = training.cv_results_table(search)
    table.to_parquet(tmp_path / "grille.parquet")
    reloaded = pd.read_parquet(tmp_path / "grille.parquet")
    assert set(reloaded["max_features"]) == {"sqrt", "0.5"}
    assert len(reloaded) == 8
    # « Aucune pondération » reste une valeur vide (un texte "nan" ne serait pas compté).
    for col in ("class_weight", "max_depth"):
        assert reloaded[col].isna().sum() == 4
