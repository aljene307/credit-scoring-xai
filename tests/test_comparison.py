"""Tests des tableaux comparatifs et de la page Modélisation (données synthétiques)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from credit_scoring import comparison, config, training
from credit_scoring.preprocessing import split_data

APP_PAGE = Path(__file__).resolve().parents[1] / "app" / "views" / "modelisation.py"
NUMERIC, CATEGORICAL = ["LIMIT_BAL", "PAY_0"], ["SEX"]


def make_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10_000, 500_000, n).astype(float),
            "PAY_0": rng.choice([-2, -1, 0, 1, 2], n).astype(float),
            "SEX": rng.choice([1, 2], n),
        }
    )
    df["DEFAULT"] = (rng.random(n) < 0.15 + 0.5 * (df["PAY_0"] >= 1)).astype(int)
    return df


@pytest.fixture(scope="module")
def summaries():
    df = make_frame()
    X, y = df[NUMERIC + CATEGORICAL], df["DEFAULT"]
    grids = {"logreg": {"C": [0.1, 1]}, "arbre": {"max_depth": [3, 5]}}
    result = {}
    for name, grid in grids.items():
        _, summary = training.train_model(name, X, y, NUMERIC, CATEGORICAL, grid=grid, n_jobs=1)
        summary["taille_octets"] = 1000
        result[name] = summary
    return result


def test_comparison_table_is_sorted_and_computes_gini(summaries):
    table = comparison.comparison_table(summaries)
    assert table["AUC"].is_monotonic_decreasing
    np.testing.assert_allclose(table["Gini"], 2 * table["AUC"] - 1)
    assert set(table["_cle"]) == {"logreg", "arbre"}


def test_thresholds_table_has_f1_and_ks_rows_per_model(summaries):
    table = comparison.thresholds_table(summaries)
    assert len(table) == 4
    assert set(table["critère"]) == {"F1 maximal (recommandé)", "KS"}
    assert table.loc[table["critère"] == "KS", "KS"].notna().all()


def test_estimate_seconds_scales_with_combinations_and_rows():
    reference = {"xgboost": {"duree_s": 240.0, "combinaisons": 80}}
    grid = {"a": [1, 2], "b": [1, 2, 3, 4, 5]}  # 10 combinaisons
    assert comparison.estimate_seconds("xgboost", grid, 24_000, reference) == pytest.approx(30.0)
    assert comparison.estimate_seconds("xgboost", grid, 12_000, reference) == pytest.approx(15.0)
    assert comparison.estimate_seconds("logreg", grid, 24_000, reference) is None


def test_default_selection_uses_reference_best_values():
    reference = {"random_forest": {"meilleurs_hyperparametres": {"max_depth": None, "min_samples_leaf": 10,
                                                                 "max_features": "sqrt"}}}
    selection = comparison.default_selection("random_forest", reference)
    assert selection == {"max_depth": [None], "min_samples_leaf": [10], "max_features": ["sqrt"]}
    assert comparison.default_selection("logreg", None) == {"C": [0.01]}


def write_reference(directory: Path, summaries: dict) -> None:
    meta = {"cible": "DEFAULT", "variables": NUMERIC + CATEGORICAL, "numeriques": NUMERIC,
            "categorielles": CATEGORICAL, "part_test": 0.2, "graine": 42, "plis": 5,
            "lignes_entrainement": 320, "versions": {"python": "3.11", "scikit-learn": "1.9.1", "xgboost": "3.2.0"}}
    training.save_results({"meta": meta, "modeles": summaries}, directory / "resultats.json")


def make_preparation() -> dict:
    df = make_frame()
    X_train, X_test, y_train, y_test = split_data(df, "DEFAULT", NUMERIC + CATEGORICAL)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "numeric": NUMERIC, "categorical": CATEGORICAL, "target": "DEFAULT",
            "class_names": {0: "Pas de défaut", 1: "Défaut"}, "test_size": 0.2, "seed": 42}


def test_page_without_reference_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    at = AppTest.from_file(str(APP_PAGE), default_timeout=60).run()
    assert not at.exception
    assert "Aucun modèle de référence" in at.warning[0].value
    assert "page « 2. Préparation »" in at.info[0].value


def test_page_shows_reference_models_and_retrains_from_the_interface(tmp_path, monkeypatch, summaries):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    write_reference(tmp_path, summaries)
    at = AppTest.from_file(str(APP_PAGE), default_timeout=120)
    at.session_state["preparation"] = make_preparation()
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 2  # tableau comparatif et seuils

    at.button[0].click().run()  # régression logistique, valeur cochée par défaut
    assert not at.exception
    trained = at.session_state["modeles_session"]
    assert list(trained) == ["logreg"]
    assert trained["logreg"]["resume"]["combinaisons"] == 1
