"""Tests de l'équité, des explications globales, des graphiques et de la page « 5. Explicabilité »."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from credit_scoring import config, explain, fairness, global_explanations, plots, reference, training
from credit_scoring.config import CATEGORICAL_FEATURES
from credit_scoring.models import MODEL_LABELS
from credit_scoring.preprocessing import split_data

APP_PAGE = Path(__file__).resolve().parents[1] / "app" / "views" / "explicabilite.py"
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]


def make_frame(n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(1, 50, n) * 10_000,
            "SEX": rng.choice([1, 2], n),
            "EDUCATION": rng.choice([1, 2, 3, 4], n),
            "MARRIAGE": rng.choice([1, 2, 3], n),
            "AGE": rng.integers(21, 70, n),
            "PAY_0": rng.choice([-2, -1, 0, 1, 2], n),
            "BILL_AMT1": rng.integers(0, 200_000, n),
            "PAY_AMT1": rng.integers(0, 20_000, n),
        },
        index=pd.Index(range(1, n + 1), name="ID"),
    )
    df["DEFAULT"] = (rng.random(n) < 0.12 + 0.5 * (df["PAY_0"] >= 1)).astype(int)
    return df


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory):
    """Quatre petits modèles, leur résumé et les explications globales, comme les scripts le feraient."""
    directory = tmp_path_factory.mktemp("models")
    df = make_frame()
    X_train, X_test, y_train, y_test = split_data(df, "DEFAULT", FEATURES)
    grids = {"logreg": {"C": [1]}, "arbre": {"max_depth": [3]},
             "random_forest": {"n_estimators": [20], "max_depth": [4]},
             "xgboost": {"n_estimators": [30], "max_depth": [2]}}
    summaries, models = {}, {}
    for name, grid in grids.items():
        search, summary = training.train_model(name, X_train, y_train, NUMERIC, CATEGORICAL_FEATURES, grid=grid, n_jobs=1)
        summary["taille_octets"] = training.save_model(search.best_estimator_, directory / f"{name}.joblib")
        summaries[name], models[name] = summary, search.best_estimator_
    meta = {"cible": "DEFAULT", "variables": FEATURES, "numeriques": NUMERIC, "categorielles": CATEGORICAL_FEATURES,
            "part_test": 0.2, "graine": 42, "plis": 5, "lignes_entrainement": len(X_train),
            "versions": {"python": "3.11", "scikit-learn": "1.9.1", "xgboost": "3.2.0"}}
    results = {"meta": meta, "modeles": summaries}
    training.save_results(results, directory / "resultats.json")
    summary = global_explanations.compute_and_save(
        directory, models, results, X_train, X_test, y_train, y_test, CATEGORICAL_FEATURES,
        n_clients=60, background_size=100, n_repeats=2, log=lambda *_: None,
    )
    return directory, df, models, X_test, y_test, summary


def test_group_errors_on_a_hand_checkable_case():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    proba = np.array([0.9, 0.1, 0.8, 0.2, 0.9, 0.9, 0.1, 0.1])
    groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    table, tests = fairness.group_errors(y, proba, 0.5, groups)
    a, b = table.set_index("groupe").loc["A"], table.set_index("groupe").loc["B"]
    assert a["rappel (%)"] == 50 and a["faux positifs (%)"] == 50
    assert b["rappel (%)"] == 100 and b["faux positifs (%)"] == pytest.approx(100 / 3)
    assert set(tests) == {"rappel", "faux positifs"}


def test_age_bands_cover_all_ages():
    bands = fairness.age_bands(pd.Series([21, 25, 26, 35, 36, 55, 56, 79]))
    assert bands.tolist() == ["21-25 ans", "21-25 ans", "26-35 ans", "26-35 ans", "36-45 ans", "46-55 ans",
                              "56 ans et +", "56 ans et +"]


def test_model_importance_is_per_original_variable(artefacts):
    _, _, models, X_test, _, _ = artefacts
    for name in MODEL_LABELS:
        importance = explain.model_importance(models[name], FEATURES, CATEGORICAL_FEATURES)
        assert set(importance.index) == set(FEATURES)
        assert (importance >= 0).all() and importance.is_monotonic_decreasing


def test_compute_and_save_writes_everything_and_reloads(artefacts):
    directory, _, _, X_test, _, summary = artefacts
    loaded = global_explanations.load_global(directory)
    assert loaded["clients_expliques"] == 60
    for name in ("xgboost", "logreg"):
        values = loaded["valeurs_shap"][name]
        assert list(values.columns) == FEATURES and len(values) == 60
        assert set(values.index) <= set(X_test.index)
    assert loaded["shap"]["xgboost"]["echelle"] == "probabilité" and loaded["shap"]["logreg"]["echelle"] == "log-odds"
    assert set(loaded["permutation"]["modele"]) == set(MODEL_LABELS)
    fair = loaded["equite"]
    assert set(fair["groupes"]) == {"sexe", "âge"} and set(fair["groupes"]["sexe"]) == {"avec", "sans"}
    assert (directory / "xgboost_sans_sensibles.joblib").exists()


def test_waterfall_ends_exactly_on_the_prediction(artefacts):
    _, _, models, X_test, _, summary = artefacts
    background = global_explanations.reference_background(models["xgboost"], X_test, 50)
    values, base = explain.ShapExplainer(models["xgboost"], background, CATEGORICAL_FEATURES).explain(X_test.iloc[:1])
    fig = plots.waterfall_figure(values.iloc[0], X_test.iloc[0], base, "probabilité", top=4)
    trace = fig.data[0]
    total = trace.x[0] + sum(trace.x[1:-1])
    assert total == pytest.approx(trace.x[-1])
    assert trace.x[-1] == pytest.approx(models["xgboost"].predict_proba(X_test.iloc[:1])[:, 1][0] * 100)
    assert trace.y[-2] == f"{len(FEATURES) - 4} autres variables"


def test_global_figures_are_built(artefacts):
    directory, _, _, X_test, _, _ = artefacts
    loaded = global_explanations.load_global(directory)
    values = loaded["valeurs_shap"]["xgboost"]
    assert len(plots.beeswarm_figure(values, X_test, "probabilité", top=5).data) == 5
    dep = plots.dependence_figure(values, X_test, "PAY_0", "probabilité")
    assert len(dep.data) == 2  # points + moyenne par valeur (variable discrète)
    assert plots.permutation_figure(loaded["permutation"], MODEL_LABELS).data
    rules = pd.Series({"PAY_0 > 0.00": 0.12, "LIMIT_BAL <= 50000.00": 0.07, "AGE > 40": -0.02})
    assert list(plots.lime_figure(rules).data[0].marker.color) == [plots.DOWN, plots.UP, plots.UP]


@pytest.fixture
def page(artefacts, monkeypatch):
    directory, df, *_ = artefacts
    monkeypatch.setattr(config, "MODELS_DIR", directory)
    monkeypatch.setattr(reference, "load_credit_data", lambda: df)
    monkeypatch.setattr(reference, "recode_undocumented", lambda frame: frame)
    return artefacts


def test_page_explains_a_client_with_both_models(page):
    _, _, models, X_test, _, _ = page
    client_id = int(X_test.index[0])
    at = AppTest.from_file(str(APP_PAGE), default_timeout=180)
    at.session_state["client_id"] = client_id
    at.run()
    assert not at.exception
    sentence = at.success[0].value
    assert sentence.startswith("Le risque de défaut est")
    proba = models["xgboost"].predict_proba(X_test.loc[[client_id]])[:, 1][0]
    assert at.metric[0].value == f"{proba:.1%}"

    at.radio[0].set_value("logreg").run()  # régression logistique : avertissement sur les log-odds
    assert not at.exception
    assert any("log-odds" in info.value for info in at.info)


def test_page_shows_fairness_and_handles_an_unknown_id(page):
    _, df, _, X_test, _, _ = page
    at = AppTest.from_file(str(APP_PAGE), default_timeout=180)
    # Identifiant d'entraînement compris entre le plus petit et le plus grand identifiant du test
    # (le champ numérique ramène sinon la valeur dans ces bornes).
    low, high = X_test.index.min(), X_test.index.max()
    train_only = next(i for i in df.index if low < i < high and i not in X_test.index)
    at.session_state["client_id"] = int(train_only)
    at.run()
    assert not at.exception
    assert "pas dans le jeu de test" in at.warning[0].value
    assert any("Erreurs selon le sexe" in m.value for m in at.markdown)
