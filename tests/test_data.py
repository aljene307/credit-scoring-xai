"""Tests du module data, sans accès réseau."""

import pandas as pd
import pytest

from credit_scoring import data
from credit_scoring.config import COLUMN_NAMES, FEATURE_NAMES, ID_COL, TARGET


def make_raw_frame(n_rows: int = 3) -> pd.DataFrame:
    """Faux tableau au format brut de l'UCI : ID, X1…X23, Y."""
    raw = pd.DataFrame({col: range(n_rows) for col in COLUMN_NAMES})
    raw[ID_COL] = range(1, n_rows + 1)
    raw["Y"] = [0, 1, 0][:n_rows]
    return raw


def make_variables(**overrides: str) -> pd.DataFrame:
    """Fausses métadonnées UCI ; overrides remplace la description d'une variable."""
    descriptions = {**FEATURE_NAMES, **overrides}
    return pd.DataFrame({"name": list(descriptions), "description": list(descriptions.values())})


def test_rename_columns_gives_readable_names():
    renamed = data.rename_columns(make_raw_frame())
    expected = list(FEATURE_NAMES.values()) + [TARGET]
    assert set(expected) <= set(renamed.columns)
    assert not any(col.startswith("X") for col in renamed.columns)
    assert len(expected) == 24


def test_rename_columns_fails_on_missing_column():
    with pytest.raises(ValueError, match="X6"):
        data.rename_columns(make_raw_frame().drop(columns="X6"))


def test_check_variable_names_accepts_expected_metadata():
    data.check_variable_names(make_variables())


def test_check_variable_names_detects_swapped_columns():
    with pytest.raises(ValueError, match="X6"):
        data.check_variable_names(make_variables(X6="PAY_2", X7="PAY_0"))


def test_load_credit_data_writes_then_reads_cache(tmp_path, monkeypatch):
    calls = []

    def fake_download():
        calls.append(1)
        return data.rename_columns(make_raw_frame()).set_index(ID_COL)

    monkeypatch.setattr(data, "download_credit_data", fake_download)
    cache = tmp_path / "sub" / "cache.parquet"

    first = data.load_credit_data(cache)
    second = data.load_credit_data(cache)

    assert cache.exists()
    assert len(calls) == 1  # le second appel lit le cache
    pd.testing.assert_frame_equal(first, second)
    assert set(second[TARGET].unique()) <= {0, 1}
    assert second.index.name == ID_COL


def make_clean_frame() -> pd.DataFrame:
    """Faux tableau aux noms lisibles, avec toutes les modalités rencontrées."""
    return pd.DataFrame(
        {
            "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
            "MARRIAGE": [0, 1, 2, 3, 1, 2, 3],
            "PAY_0": [-2, -1, 0, 1, 2, 8, 0],
            "LIMIT_BAL": [10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000],
        }
    )


def test_recode_undocumented_groups_education_codes():
    recoded = data.recode_undocumented(make_clean_frame())
    assert recoded["EDUCATION"].tolist() == [4, 1, 2, 3, 4, 4, 4]
    assert set(recoded["EDUCATION"]) == {1, 2, 3, 4}


def test_recode_undocumented_groups_marriage_zero():
    recoded = data.recode_undocumented(make_clean_frame())
    assert recoded["MARRIAGE"].tolist() == [3, 1, 2, 3, 1, 2, 3]


def test_recode_undocumented_leaves_other_columns_and_input_unchanged():
    original = make_clean_frame()
    snapshot = original.copy()
    recoded = data.recode_undocumented(original)
    pd.testing.assert_frame_equal(original, snapshot)  # pas de modification sur place
    pd.testing.assert_series_equal(recoded["PAY_0"], original["PAY_0"])
    pd.testing.assert_series_equal(recoded["LIMIT_BAL"], original["LIMIT_BAL"])


def test_load_credit_data_refresh_downloads_again(tmp_path, monkeypatch):
    calls = []

    def fake_download():
        calls.append(1)
        return data.rename_columns(make_raw_frame()).set_index(ID_COL)

    monkeypatch.setattr(data, "download_credit_data", fake_download)
    cache = tmp_path / "cache.parquet"
    data.load_credit_data(cache)
    data.load_credit_data(cache, refresh=True)
    assert len(calls) == 2
