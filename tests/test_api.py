"""Tests de l'API FastAPI et du client utilisé par l'application (données synthétiques)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import app
from api.schemas import EXEMPLE_CLIENT
from credit_scoring import api_client, config, global_explanations, reference, training
from credit_scoring.config import CATEGORICAL_FEATURES, FEATURE_NAMES, SENSITIVE_FEATURES
from credit_scoring.preprocessing import split_data

FEATURES = list(FEATURE_NAMES.values())
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
API_PAGE = Path(__file__).resolve().parents[1] / "app" / "views" / "api_demo.py"


def make_frame(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {f: rng.integers(0, 50_000, n) for f in FEATURES}
    data.update({
        "LIMIT_BAL": rng.integers(1, 50, n) * 10_000, "SEX": rng.choice([1, 2], n),
        "EDUCATION": rng.choice([1, 2, 3, 4], n), "MARRIAGE": rng.choice([1, 2, 3], n), "AGE": rng.integers(21, 70, n),
        **{f: rng.choice([-2, -1, 0, 1, 2], n) for f in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]},
    })
    df = pd.DataFrame(data, index=pd.Index(range(1, n + 1), name="ID"))[FEATURES]
    df["DEFAULT"] = (rng.random(n) < 0.12 + 0.5 * (df["PAY_0"] >= 1)).astype(int)
    return df


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("models")
    df = make_frame()
    X_train, X_test, y_train, y_test = split_data(df, "DEFAULT", FEATURES)
    grids = {"logreg": {"C": [1]}, "xgboost": {"n_estimators": [30], "max_depth": [2]}}
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
    global_explanations.compute_and_save(directory, models, results, X_train, X_test, y_train, y_test,
                                         CATEGORICAL_FEATURES, n_clients=40, background_size=100, n_repeats=2,
                                         log=lambda *_: None)
    return directory, df, models, X_test


@pytest.fixture
def api(models_dir, monkeypatch):
    directory, df, models, X_test = models_dir
    monkeypatch.setattr(config, "MODELS_DIR", directory)
    monkeypatch.setattr(reference, "load_credit_data", lambda: df)
    monkeypatch.setattr(reference, "recode_undocumented", lambda frame: frame)
    with TestClient(app) as client:
        yield client, models, X_test


def client_of(X_test, i=0) -> dict:
    return {k: int(v) for k, v in X_test.iloc[i].items()}


def test_health_and_model_information(api):
    client, _, _ = api
    health = client.get("/sante").json()
    assert health["statut"] == "ok" and health["modeles_disponibles"] == ["complet", "sans_sensibles"]
    info = client.get("/modele", params={"modele": "sans_sensibles"}).json()
    assert sorted(info["variables_ignorees"]) == sorted(SENSITIVE_FEATURES)
    assert info["variables"]["SEX"]["utilisee"] is False and info["variables"]["PAY_0"]["utilisee"] is True
    assert 0 < info["seuil_recommande"] < 1 and "hors pli" in info["methode_seuil"]


def test_prediction_matches_the_model_loaded_directly(api):
    client, models, X_test = api
    body = client_of(X_test)
    response = client.post("/predire", json={"client": body, "modele": "complet"}).json()
    expected = models["xgboost"].predict_proba(X_test.iloc[[0]])[:, 1][0]
    assert response["probabilite"] == pytest.approx(expected, abs=1e-7)
    assert response["variables_ignorees"] == []
    assert response["signale"] == (response["probabilite"] >= response["seuil"])


def test_model_without_sensitive_variables_ignores_them_and_says_so(api):
    client, _, X_test = api
    body = client_of(X_test)
    changed = {**body, "SEX": 3 - body["SEX"], "AGE": 65, "MARRIAGE": 3, "EDUCATION": 4}
    a = client.post("/predire", json={"client": body, "modele": "sans_sensibles"}).json()
    b = client.post("/predire", json={"client": changed, "modele": "sans_sensibles"}).json()
    assert a["probabilite"] == b["probabilite"]
    assert sorted(a["variables_ignorees"]) == sorted(SENSITIVE_FEATURES)


def test_default_model_is_the_one_recommended_for_deployment(api):
    client, _, X_test = api
    assert client.post("/predire", json={"client": client_of(X_test)}).json()["modele"] == "sans_sensibles"


def test_custom_threshold_changes_the_decision(api):
    client, _, X_test = api
    low = client.post("/predire", json={"client": client_of(X_test), "modele": "complet", "seuil": 0.001}).json()
    high = client.post("/predire", json={"client": client_of(X_test), "modele": "complet", "seuil": 0.999}).json()
    assert low["signale"] is True and high["signale"] is False


@pytest.mark.parametrize("change", [{"SEX": 7}, {"AGE": -3}, {"PAY_0": 9}, {"inconnue": 1}])
def test_invalid_clients_are_refused(api, change):
    client, _, _ = api
    response = client.post("/predire", json={"client": {**EXEMPLE_CLIENT, **change}})
    assert response.status_code == 422


def test_missing_variable_is_refused(api):
    client, _, _ = api
    body = {k: v for k, v in EXEMPLE_CLIENT.items() if k != "LIMIT_BAL"}
    assert client.post("/predire", json={"client": body}).status_code == 422


def test_explanation_adds_up_and_contains_both_methods(api):
    client, _, X_test = api
    e = client.post("/expliquer", json={"client": client_of(X_test), "modele": "complet"}).json()
    assert e["valeur_de_base"] + sum(e["contributions_shap"].values()) == pytest.approx(e["probabilite"], abs=1e-5)
    assert len(e["principales_shap"]) == 5 and len(e["regles_lime"]) == 5
    assert e["accord"]["communes"] <= 5 and e["phrase"].startswith("Le risque de défaut est")
    only_shap = client.post("/expliquer", json={"client": client_of(X_test), "modele": "complet",
                                                "methodes": ["shap"]}).json()
    assert only_shap["regles_lime"] is None and only_shap["accord"] is None


def test_every_example_shown_in_docs_is_accepted(api):
    """Régression : sans exemple complet, /docs fabriquait une valeur de seuil refusée par l'API."""
    client, _, _ = api
    paths = client.get("/openapi.json").json()["paths"]
    sent = 0
    for path, methods in paths.items():
        if "post" not in methods:
            continue
        examples = methods["post"]["requestBody"]["content"]["application/json"].get("examples", {})
        assert examples, f"{path} n'a pas d'exemple complet dans /docs"
        for name, example in examples.items():
            response = client.post(path, json=example["value"])
            assert response.status_code == 200, f"{path}, exemple {name} : {response.text[:200]}"
            sent += 1
    assert sent == 5  # 3 exemples pour /predire, 2 pour /expliquer


def test_global_explanations(api):
    client, _, _ = api
    g = client.get("/explications/globales").json()
    assert g["modele"] == "complet" and g["clients_expliques"] == 40
    assert set(g["permutation"]) == {"logreg", "xgboost"}
    assert list(g["contribution_moyenne"]) == sorted(g["contribution_moyenne"], key=g["contribution_moyenne"].get,
                                                     reverse=True)


def test_api_starts_without_the_dataset(models_dir, monkeypatch):
    """Paquet de démarrage (choix API05) : aucune lecture ni téléchargement du jeu de données."""
    directory, _, models, X_test = models_dir
    monkeypatch.setattr(config, "MODELS_DIR", directory)

    def forbidden(*args, **kwargs):
        raise AssertionError("l'API ne doit pas charger le jeu de données")

    monkeypatch.setattr(reference, "load_credit_data", forbidden)
    monkeypatch.setattr(reference, "reference_split", forbidden)
    with TestClient(app) as client:
        response = client.post("/expliquer", json={"client": client_of(X_test), "modele": "complet"})
    assert response.status_code == 200
    expected = models["xgboost"].predict_proba(X_test.iloc[[0]])[:, 1][0]
    assert response.json()["probabilite"] == pytest.approx(expected, abs=1e-7)


def test_api_answers_503_when_models_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    with TestClient(app) as client:
        response = client.get("/sante")
    assert response.status_code == 503 and "scripts/train.py" in response.json()["detail"]


def test_dotenv_values_are_cleaned(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b'# commentaire\r\nAPI_URL =  "http://127.0.0.1:8000/"  \r\nAPI_TIMEOUT=5\r\n')
    values = api_client.read_dotenv(env)
    assert values == {"API_URL": "http://127.0.0.1:8000/", "API_TIMEOUT": "5"}


def test_client_raises_when_the_api_is_unreachable():
    with pytest.raises(api_client.ApiIndisponible):
        api_client.sante(timeout=0.5)


def test_api_page_explains_how_to_start_the_api_when_it_is_down():
    at = AppTest.from_file(str(API_PAGE), default_timeout=60).run()
    assert not at.exception
    assert "ne répond pas" in at.error[0].value
    assert "uvicorn api.main:app" in at.code[0].value
