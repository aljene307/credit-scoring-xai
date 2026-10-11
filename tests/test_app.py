"""Exécution réelle des pages Streamlit avec AppTest (sans navigateur)."""

from pathlib import Path

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
