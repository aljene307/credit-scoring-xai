"""Tests des calculs exploratoires, des graphiques et de la préparation d'un CSV importé."""

import numpy as np
import pandas as pd
import pytest

from credit_scoring import eda, plots
from credit_scoring.data import binarize_target

CLASS_NAMES = {0: "Pas de défaut", 1: "Défaut"}


def make_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "SEX": rng.choice([1, 2], n),
            "PAY_0": rng.choice([-2, -1, 0, 1, 2], n),
            "LIMIT_BAL": rng.integers(10_000, 500_000, n),
            "DEFAULT": rng.choice([0, 1], n, p=[0.78, 0.22]),
        }
    )


def test_missing_summary_lists_only_columns_with_missing_values():
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    summary = eda.missing_summary(df)
    assert list(summary.index) == ["a"]
    assert summary.loc["a", "manquantes"] == 2
    assert summary.loc["a", "part (%)"] == 50.0


def test_class_balance_counts_and_shares():
    balance = eda.class_balance(pd.Series([0, 0, 0, 1]))
    assert balance["clients"].tolist() == [3, 1]
    assert balance["part (%)"].tolist() == [75.0, 25.0]


def test_display_value_uses_labels():
    assert eda.display_value("SEX", 2) == "Femme"
    assert eda.display_value("EDUCATION", 4) == "Autres ou inconnu"
    assert eda.display_value("PAY_0", -2) == "-2 : Pas d'utilisation"
    assert eda.display_value("AUTRE", 7) == "7"


def test_positive_rate_by_categorical_matches_manual_computation():
    df = make_frame()
    table = eda.positive_rate_by(df, "SEX", "DEFAULT", categorical=True)
    assert list(table.index) == ["Homme", "Femme"]
    expected = round(df.loc[df["SEX"] == 2, "DEFAULT"].mean() * 100, 1)
    assert table.loc["Femme", "taux (%)"] == expected
    assert table["clients"].sum() == len(df)


def test_positive_rate_by_few_values_is_per_value_in_order():
    table = eda.positive_rate_by(make_frame(), "PAY_0", "DEFAULT", categorical=False)
    assert [label.split(" : ")[0] for label in table.index] == ["-2", "-1", "0", "1", "2"]


def test_positive_rate_by_continuous_uses_quantile_bins():
    df = make_frame()
    table = eda.positive_rate_by(df, "LIMIT_BAL", "DEFAULT", categorical=False, n_bins=5)
    assert len(table) == 5
    assert table["clients"].sum() == len(df)


def test_quantile_bins_are_labelled_with_observed_ranges():
    # 20 valeurs distinctes : au-delà du seuil, donc découpage en tranches.
    df = pd.DataFrame({"AGE": range(21, 41), "LIMIT_BAL": range(10_000, 210_000, 10_000), "y": [0, 1] * 10})
    ages = eda.positive_rate_by(df, "AGE", "y", categorical=False, n_bins=2)
    assert list(ages.index) == ["21 – 30", "31 – 40"]
    limits = eda.positive_rate_by(df, "LIMIT_BAL", "y", categorical=False, n_bins=2)
    assert list(limits.index) == ["10 000 – 100 000", "110 000 – 200 000"]


def test_positive_rate_by_keeps_missing_values_as_a_group():
    df = make_frame()
    df["LIMIT_BAL"] = df["LIMIT_BAL"].astype(float)
    df.loc[:9, "LIMIT_BAL"] = np.nan
    table = eda.positive_rate_by(df, "LIMIT_BAL", "DEFAULT", categorical=False, n_bins=4)
    assert table.index[-1] == "manquant"
    assert table.loc["manquant", "clients"] == 10
    assert table["clients"].sum() == len(df)


def test_text_column_is_treated_as_discrete_even_if_not_declared():
    df = pd.DataFrame({"ville": ["A", "B", "A", "C"], "y": [1, 0, 0, 1]})
    assert eda.is_discrete(df["ville"], categorical=False)
    table = eda.positive_rate_by(df, "ville", "y", categorical=False)
    assert list(table.index) == ["A", "B", "C"]


def test_binarize_target_maps_positive_to_one_and_drops_missing():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "churn": ["oui", "non", None, "oui"]})
    prepared = binarize_target(df, "churn", "oui")
    assert prepared["churn"].tolist() == [1, 0, 1]
    assert len(prepared) == 3


def test_binarize_target_rejects_non_binary_target():
    df = pd.DataFrame({"y": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="exactement deux valeurs"):
        binarize_target(df, "y", "a")


def test_figures_are_built_for_each_kind_of_variable():
    df = make_frame()
    assert plots.class_balance_figure(eda.class_balance(df["DEFAULT"]), CLASS_NAMES).data
    for col, categorical in [("SEX", True), ("PAY_0", False), ("LIMIT_BAL", False)]:
        for normalize in (True, False):
            fig = plots.distribution_figure(df, col, "DEFAULT", categorical, CLASS_NAMES, normalize)
            assert len(fig.data) == 2  # une trace par classe
        table = eda.positive_rate_by(df, col, "DEFAULT", categorical)
        assert plots.positive_rate_figure(table, col, 22.0, "Défaut").data
    corr = df.corr()
    assert plots.correlation_figure(corr, "Pearson").data
