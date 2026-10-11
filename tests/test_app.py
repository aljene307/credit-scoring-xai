"""Exécution réelle des pages Streamlit avec AppTest (sans navigateur)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from credit_scoring.config import CACHE_FILE

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def test_home_page_runs():
    at = AppTest.from_file(str(APP_DIR / "views" / "accueil.py")).run()
    assert not at.exception
    assert at.title[0].value == "Scoring de crédit explicable"


@pytest.mark.skipif(not CACHE_FILE.exists(), reason="cache des données absent (pas de réseau dans les tests)")
def test_data_page_runs_on_taiwan_dataset_and_every_tab():
    at = AppTest.from_file(str(APP_DIR / "views" / "donnees.py"), default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].value == "30 000"
    assert at.metric[3].value == "22.1 %"
    assert at.session_state["dataset"]["target"] == "DEFAULT"

    # Changer de variable dans les onglets Distributions et Taux par variable.
    for key, variable in [("dist_var", "EDUCATION"), ("rate_var", "PAY_0"), ("rate_var", "LIMIT_BAL")]:
        at.selectbox(key=key).select(variable).run()
        assert not at.exception

    at.radio[1].set_value("Spearman").run()  # méthode des corrélations
    assert not at.exception


def make_dataset(n: int = 500) -> dict:
    """Jeu synthétique au format de st.session_state["dataset"] (pas de réseau)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10_000, 500_000, n),
            "SEX": rng.choice([1, 2], n),
            "AGE": rng.integers(21, 79, n),
            "PAY_0": rng.choice([-2, -1, 0, 1, 2], n),
            "DEFAULT": rng.choice([0, 1], n, p=[0.78, 0.22]),
        }
    )
    return {
        "df": df,
        "target": "DEFAULT",
        "categorical": ["SEX"],
        "class_names": {0: "Pas de défaut", 1: "Défaut"},
        "source": "jeu synthétique de test",
    }


def test_preparation_page_asks_for_data_first():
    at = AppTest.from_file(str(APP_DIR / "views" / "preparation.py")).run()
    assert not at.exception
    assert "page « 1. Données »" in at.info[0].value


def test_preparation_page_splits_and_stores_the_result():
    at = AppTest.from_file(str(APP_DIR / "views" / "preparation.py"), default_timeout=60)
    at.session_state["dataset"] = make_dataset()
    at.run()
    assert not at.exception
    prep = at.session_state["preparation"]
    assert len(prep["X_test"]) == 100 and len(prep["X_train"]) == 400
    assert prep["categorical"] == ["SEX"]
    assert prep["numeric"] == ["LIMIT_BAL", "AGE", "PAY_0"]

    # Exclure les variables sensibles : SEX et AGE disparaissent.
    at.checkbox[0].check().run()
    assert not at.exception
    assert list(at.session_state["preparation"]["X_train"].columns) == ["LIMIT_BAL", "PAY_0"]

    # Changer la part du test.
    at.slider[0].set_value(30).run()
    assert len(at.session_state["preparation"]["X_test"]) == 150
