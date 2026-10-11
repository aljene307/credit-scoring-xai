"""Tests de la séparation et du prétraitement."""

import numpy as np
import pandas as pd
import pytest

from credit_scoring.preprocessing import build_preprocessor, split_data, split_feature_types


def make_frame(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10_000, 500_000, n).astype(float),
            "AGE": rng.integers(21, 79, n).astype(float),
            "EDUCATION": rng.choice([1, 2, 3, 4], n),
            "DEFAULT": rng.choice([0, 1], n, p=[0.78, 0.22]),
        }
    )


NUMERIC, CATEGORICAL = ["LIMIT_BAL", "AGE"], ["EDUCATION"]
FEATURES = NUMERIC + CATEGORICAL


def test_split_is_stratified_and_sized():
    df = make_frame()
    X_train, X_test, y_train, y_test = split_data(df, "DEFAULT", FEATURES, test_size=0.2)
    assert len(X_test) == 200 and len(X_train) == 800
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert set(X_train.index).isdisjoint(X_test.index)


def test_split_is_reproducible_with_the_same_seed():
    df = make_frame()
    first = split_data(df, "DEFAULT", FEATURES, random_state=42)[1].index
    second = split_data(df, "DEFAULT", FEATURES, random_state=42)[1].index
    other = split_data(df, "DEFAULT", FEATURES, random_state=7)[1].index
    assert first.equals(second)
    assert not first.equals(other)


def test_text_columns_are_treated_as_categorical():
    df = pd.DataFrame({"a": [1.0, 2.0], "ville": ["A", "B"], "code": [1, 2]})
    numeric, categorical = split_feature_types(df, ["a", "ville", "code"], categorical=["code"])
    assert numeric == ["a"]
    assert categorical == ["ville", "code"]


def test_scaled_preprocessor_learns_only_on_training_data():
    df = make_frame()
    X_train, X_test, _, _ = split_data(df, "DEFAULT", FEATURES)
    X_test = X_test.copy()
    X_test["LIMIT_BAL"] += 1_000_000  # jeu de test volontairement décalé

    pre = build_preprocessor(NUMERIC, CATEGORICAL, scale=True).fit(X_train)
    train_out, test_out = pre.transform(X_train), pre.transform(X_test)

    assert train_out["LIMIT_BAL"].mean() == pytest.approx(0, abs=1e-9)
    assert train_out["LIMIT_BAL"].std(ddof=0) == pytest.approx(1)
    # Le test est transformé avec la moyenne et l'écart-type de l'entraînement.
    mean, std = X_train["LIMIT_BAL"].mean(), X_train["LIMIT_BAL"].std(ddof=0)
    expected = (X_test["LIMIT_BAL"] - mean) / std
    np.testing.assert_allclose(test_out["LIMIT_BAL"], expected)
    assert test_out["LIMIT_BAL"].mean() > 5


def test_unscaled_preprocessor_keeps_original_units():
    df = make_frame()
    pre = build_preprocessor(NUMERIC, CATEGORICAL, scale=False)
    out = pre.fit_transform(df[FEATURES])
    pd.testing.assert_series_equal(out["LIMIT_BAL"], df["LIMIT_BAL"])
    pd.testing.assert_series_equal(out["AGE"], df["AGE"])


def test_one_hot_columns_are_named_and_unknown_category_is_ignored():
    df = make_frame()
    pre = build_preprocessor(NUMERIC, CATEGORICAL, scale=False).fit(df[FEATURES])
    names = list(pre.get_feature_names_out())
    assert names == ["LIMIT_BAL", "AGE", "EDUCATION_1", "EDUCATION_2", "EDUCATION_3", "EDUCATION_4"]

    new = pd.DataFrame({"LIMIT_BAL": [50_000.0], "AGE": [30.0], "EDUCATION": [9]})
    out = pre.transform(new)
    assert out.filter(like="EDUCATION_").sum(axis=1).iloc[0] == 0  # modalité inconnue : que des zéros


def test_missing_values_are_imputed_with_training_statistics():
    train = pd.DataFrame({"LIMIT_BAL": [10.0, 20.0, 30.0, 1000.0], "AGE": [20.0, 30.0, 40.0, 50.0],
                          "EDUCATION": [1, 1, 2, 1]})
    pre = build_preprocessor(NUMERIC, CATEGORICAL, scale=False).fit(train)
    new = pd.DataFrame({"LIMIT_BAL": [np.nan], "AGE": [np.nan], "EDUCATION": [np.nan]})
    out = pre.transform(new)
    assert out["LIMIT_BAL"].iloc[0] == 25.0  # médiane, pas la moyenne (265)
    assert out["AGE"].iloc[0] == 35.0
    assert out["EDUCATION_1"].iloc[0] == 1.0  # modalité la plus fréquente
