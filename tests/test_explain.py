"""Tests du module d'explicabilité (données synthétiques, petits modèles)."""

import numpy as np
import pandas as pd
import pytest

from credit_scoring import explain, training

NUMERIC, CATEGORICAL = ["LIMIT_BAL", "PAY_0"], ["SEX"]
FEATURES = NUMERIC + CATEGORICAL


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10_000, 500_000, n).astype(float),
            "PAY_0": rng.choice([-2, -1, 0, 1, 2], n).astype(float),
            "SEX": rng.choice([1, 2], n),
        }
    )
    y = pd.Series((rng.random(n) < 0.15 + 0.5 * (X["PAY_0"] >= 1)).astype(int))
    return X, y


@pytest.fixture(scope="module")
def pipelines(data):
    X, y = data
    grids = {"xgboost": {"n_estimators": [30], "max_depth": [2]}, "logreg": {"C": [1]}}
    return {name: training.train_model(name, X, y, NUMERIC, CATEGORICAL, grid=g, n_jobs=1)[0].best_estimator_
            for name, g in grids.items()}


def test_aggregate_onehot_sums_categories_and_keeps_row_totals():
    values = pd.DataFrame({"LIMIT_BAL": [0.1, -0.2], "PAY_0": [0.3, 0.0], "SEX_1": [0.05, -0.01], "SEX_2": [0.02, 0.03]})
    out = explain.aggregate_onehot(values, FEATURES, CATEGORICAL)
    assert list(out.columns) == FEATURES
    np.testing.assert_allclose(out["SEX"], [0.07, 0.02])
    np.testing.assert_allclose(out.sum(axis=1), values.sum(axis=1))


@pytest.mark.parametrize("name, method, scale", [("xgboost", "tree", "probabilité"), ("logreg", "linear", "log-odds")])
def test_shap_values_add_up_to_the_model_output(data, pipelines, name, method, scale):
    X, _ = data
    explainer = explain.ShapExplainer(pipelines[name], explain.background_sample(X, 50), CATEGORICAL)
    assert (explainer.method, explainer.scale) == (method, scale)
    values, base = explainer.explain(X.iloc[:10])
    assert list(values.columns) == FEATURES
    proba = pipelines[name].predict_proba(X.iloc[:10])[:, 1]
    expected = proba if scale == "probabilité" else np.log(proba / (1 - proba))
    np.testing.assert_allclose(base + values.sum(axis=1), expected, atol=1e-5)


def test_tree_explainer_keeps_the_whole_background(data, pipelines):
    """Régression : SHAP réduisait en silence l'échantillon de référence à 100 clients."""
    X, _ = data
    background = explain.background_sample(X, 300)
    explainer = explain.ShapExplainer(pipelines["xgboost"], background, CATEGORICAL)
    _, base = explainer.explain(X.iloc[:3])
    expected = pipelines["xgboost"].predict_proba(background)[:, 1].mean()
    assert base == pytest.approx(expected, abs=1e-6)


def test_stratified_background_spreads_clients_over_score_deciles(data):
    X, _ = data
    scores = np.linspace(0, 1, len(X))
    bg = explain.background_sample(X, 100, seed=0, scores=scores)
    assert len(bg) == 100
    deciles = pd.qcut(pd.Series(scores, index=X.index), 10, labels=False)
    assert deciles.loc[bg.index].value_counts().tolist() == [10] * 10


def test_lime_returns_signed_weights_for_original_variables(data, pipelines):
    X, _ = data
    lime = explain.LimeExplainer(X, CATEGORICAL)
    weights, rules = lime.explain(pipelines["xgboost"], X.iloc[0], num_samples=500, seed=0, num_features=3)
    assert set(weights.index) <= set(FEATURES) and len(weights) == 3
    assert weights.abs().is_monotonic_decreasing
    assert len(rules) == 3
    again, _ = lime.explain(pipelines["xgboost"], X.iloc[0], num_samples=500, seed=0, num_features=3)
    pd.testing.assert_series_equal(weights, again)  # même graine : même explication


def test_agreement_counts_common_variables_and_signs():
    shap_row = pd.Series({"A": 0.5, "B": -0.3, "C": 0.2, "D": 0.1, "E": 0.05, "F": 0.01})
    lime_w = pd.Series({"A": 0.4, "B": 0.2, "F": 0.3, "G": 0.2, "C": -0.1})
    result = explain.agreement(shap_row, lime_w, k=5)
    assert result == {"communes": 3, "meme_signe": 1, "variables": ["A", "B", "C"]}


def test_describe_uses_plain_words():
    assert explain.describe("PAY_0", 2) == "un retard de paiement de 2 mois en septembre"
    assert explain.describe("PAY_2", -2) == "aucune utilisation de la carte en août"
    assert explain.describe("LIMIT_BAL", 20000, median_limit=140000) == "un plafond de crédit faible (20 000 NT$)"
    assert explain.describe("SEX", 2) == "son sexe (femme)"
    assert explain.describe("PAY_AMT1", 3000) == "un paiement de 3 000 NT$ en septembre"


def test_summary_sentence_high_risk_groups_months_and_elides():
    contributions = pd.Series({"PAY_0": 0.25, "PAY_2": 0.10, "LIMIT_BAL": 0.08, "AGE": -0.01})
    client = pd.Series({"PAY_0": 2, "PAY_2": 2, "LIMIT_BAL": 20000, "AGE": 22})
    sentence = explain.summary_sentence(contributions, client, proba=0.715, base=0.22, threshold=0.27,
                                        median_limit=140000)
    assert sentence == (
        "Le risque de défaut est élevé (72 % contre 22 % en moyenne), principalement à cause "
        "d'un retard de paiement de 2 mois en septembre et en août et d'un plafond de crédit faible (20 000 NT$)."
    )


def test_summary_sentence_lists_grouped_months_in_calendar_order():
    # PAY_3 (juillet) contribue plus que PAY_2 (août) : l'ordre affiché reste août puis juillet.
    contributions = pd.Series({"PAY_3": 0.20, "LIMIT_BAL": 0.08, "PAY_2": 0.05})
    client = pd.Series({"PAY_3": 2, "PAY_2": 2, "LIMIT_BAL": 20000})
    sentence = explain.summary_sentence(contributions, client, proba=0.4, base=0.22, threshold=0.27)
    assert "un retard de paiement de 2 mois en août et en juillet" in sentence


def test_summary_sentence_low_and_moderate_risk():
    contributions = pd.Series({"PAY_0": -0.06, "LIMIT_BAL": -0.04, "AGE": 0.01})
    client = pd.Series({"PAY_0": -1, "LIMIT_BAL": 500000, "AGE": 40})
    low = explain.summary_sentence(contributions, client, proba=0.08, base=0.22, threshold=0.27,
                                   median_limit=140000)
    assert low == ("Le risque de défaut est faible (8 % contre 22 % en moyenne), principalement grâce "
                   "à un paiement en totalité en septembre et à un plafond de crédit élevé (500 000 NT$).")
    moderate = explain.summary_sentence(-contributions, client, proba=0.25, base=0.22, threshold=0.27)
    assert moderate.startswith("Le risque de défaut est modéré (25 % contre 22 % en moyenne)")
