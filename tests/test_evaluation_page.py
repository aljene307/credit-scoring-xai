"""Tests des graphiques d'évaluation et de la page « 4. Évaluation » (données synthétiques)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from credit_scoring import comparison, config, evaluation, plots, reference, training
from credit_scoring.preprocessing import split_data

APP_PAGE = Path(__file__).resolve().parents[1] / "app" / "views" / "evaluation.py"
NUMERIC, CATEGORICAL = ["LIMIT_BAL", "PAY_0"], ["SEX"]
FEATURES = NUMERIC + CATEGORICAL


def make_frame(n: int = 500) -> pd.DataFrame:
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
def trained(tmp_path_factory):
    """Deux modèles de « référence » entraînés et sauvegardés comme le fait scripts/train.py."""
    directory = tmp_path_factory.mktemp("models")
    df = make_frame()
    X_train, _, y_train, _ = split_data(df, "DEFAULT", FEATURES)
    summaries = {}
    for name, grid in {"logreg": {"C": [1]}, "xgboost": {"n_estimators": [30], "max_depth": [2]}}.items():
        search, summary = training.train_model(name, X_train, y_train, NUMERIC, CATEGORICAL, grid=grid, n_jobs=1)
        summary["taille_octets"] = training.save_model(search.best_estimator_, directory / f"{name}.joblib")
        summaries[name] = summary
    meta = {"cible": "DEFAULT", "variables": FEATURES, "numeriques": NUMERIC, "categorielles": CATEGORICAL,
            "part_test": 0.2, "graine": 42, "plis": 5, "lignes_entrainement": len(X_train),
            "versions": {"python": "3.11", "scikit-learn": "1.9.1", "xgboost": "3.2.0"}}
    training.save_results({"meta": meta, "modeles": summaries}, directory / "resultats.json")
    return directory, df


def test_ks_figure_shows_exactly_the_ks_of_the_table():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.3 * y + rng.random(2000) * 0.7, 0, 1)
    fig = plots.ks_figure(y, proba, {0: "Pas de défaut", 1: "Défaut"})
    shown = float(fig.data[2].name.split("KS = ")[1].split(" ")[0])
    assert shown == pytest.approx(evaluation.ks_threshold(y, proba)[1], abs=5e-4)


def test_evaluation_figures_are_built():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    probas = {"xgboost": rng.random(300), "logreg": rng.random(300)}
    labels = {"xgboost": "XGBoost", "logreg": "Régression logistique"}
    thresholds = {"xgboost": 0.5, "logreg": 0.4}
    assert len(plots.roc_figure(y, probas, labels, thresholds).data) == 5  # hasard + 2 × (courbe, point)
    assert len(plots.precision_recall_figure(y, probas, labels, thresholds).data) == 5
    assert plots.confusion_figure({"vn": 10, "fp": 2, "fn": 3, "vp": 5}).data
    assert plots.calibration_figure(evaluation.calibration_table(y, probas["xgboost"]), "XGBoost").data


def test_page_without_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    at = AppTest.from_file(str(APP_PAGE), default_timeout=60).run()
    assert not at.exception
    assert "Aucun modèle à évaluer" in at.warning[0].value


def test_page_evaluates_reference_models_and_updates_with_the_slider(trained, monkeypatch):
    directory, df = trained
    monkeypatch.setattr(config, "MODELS_DIR", directory)
    monkeypatch.setattr(reference, "load_credit_data", lambda: df)
    monkeypatch.setattr(reference, "recode_undocumented", lambda frame: frame)

    at = AppTest.from_file(str(APP_PAGE), default_timeout=120).run()
    assert not at.exception
    metrics = at.dataframe[0].value
    assert list(metrics["modèle"]) == sorted(metrics["modèle"], key=lambda m: -metrics.set_index("modèle").loc[m, "AUC test"])
    assert len(metrics) == 2
    sentence_before = at.success[0].value
    assert sentence_before.startswith("Sur 100 clients, le modèle en signale")

    # Au premier affichage, la phrase utilise le seuil recommandé exact (pas son arrondi).
    results, models = reference.load_reference(directory)
    _, X_test, _, y_test = reference.reference_split(results["meta"], df)
    exact = results["modeles"]["xgboost"]["seuils"]["f1_max"]["seuil"]
    counts = evaluation.confusion_counts(y_test, models["xgboost"].predict_proba(X_test)[:, 1], exact)
    assert sentence_before == evaluation.decision_sentence(counts)

    at.slider[0].set_value(0.05).run()  # seuil très bas : presque tous les clients sont signalés
    assert not at.exception
    assert at.success[0].value != sentence_before
    assert "Il manque 0 défauts" in at.success[0].value or "Il manque 1 défauts" in at.success[0].value


def test_holdout_table_matches_direct_computation(trained, monkeypatch):
    directory, df = trained
    results, models = reference.load_reference(directory)
    _, X_test, _, y_test = reference.reference_split(results["meta"], df)
    probas = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}
    boot = evaluation.bootstrap_auc(y_test, probas, n_resamples=50)
    table = comparison.holdout_table(y_test, probas, results["modeles"], boot)
    from sklearn.metrics import roc_auc_score
    for _, row in table.iterrows():
        assert row["AUC test"] == pytest.approx(roc_auc_score(y_test, probas[row["_cle"]]))
    diffs = comparison.auc_differences(boot, results["modeles"])
    assert list(diffs["comparaison"]) == ["XGBoost − Régression logistique"]
