"""Appels de l'application à l'API FastAPI (module 6, choix API03).

L'adresse de l'API vient, dans l'ordre : de la variable d'environnement API_URL, du fichier
.env à la racine du projet, puis de la valeur par défaut http://127.0.0.1:8000.
Si l'API ne répond pas, les fonctions lèvent ApiIndisponible : l'application calcule alors
le résultat elle-même et l'indique.
"""

import os
import time

import requests

from credit_scoring.config import PROJECT_ROOT

DEFAULT_URL = "http://127.0.0.1:8000"


class ApiIndisponible(Exception):
    """L'API n'a pas répondu, ou a répondu par une erreur."""


def read_dotenv(path=PROJECT_ROOT / ".env") -> dict[str, str]:
    """Lit un fichier .env simple (CLE=valeur). Espaces, guillemets et retours chariot autour
    des valeurs sont retirés (erreur fréquente quand une valeur est collée)."""
    values = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'").strip()
    return values


def setting(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or read_dotenv().get(name, "") or default


def api_url() -> str:
    return setting("API_URL", DEFAULT_URL).rstrip("/")


def _timeout() -> float:
    try:
        return float(setting("API_TIMEOUT", "10"))
    except ValueError:
        return 10.0


def _call(method: str, path: str, timeout: float | None = None, **kwargs) -> tuple[dict, float]:
    """Requête vers l'API ; renvoie la réponse JSON et la durée totale (ms) vue par l'application."""
    start = time.perf_counter()
    try:
        response = requests.request(method, api_url() + path, timeout=timeout or _timeout(), **kwargs)
    except requests.RequestException as exc:
        raise ApiIndisponible(f"pas de réponse de {api_url()} ({type(exc).__name__})") from exc
    if response.status_code != 200:
        raise ApiIndisponible(f"erreur {response.status_code} : {response.text[:200]}")
    return response.json(), (time.perf_counter() - start) * 1000


def sante(timeout: float = 2.0) -> dict:
    return _call("GET", "/sante", timeout=timeout)[0]


def informations(modele: str) -> dict:
    return _call("GET", "/modele", params={"modele": modele})[0]


def predire(client: dict, modele: str) -> tuple[dict, float]:
    return _call("POST", "/predire", json={"client": client, "modele": modele})


def expliquer(client: dict, modele: str, methodes=("shap", "lime"), nombre_variables: int = 5) -> tuple[dict, float]:
    payload = {"client": client, "modele": modele, "methodes": list(methodes), "nombre_variables": nombre_variables}
    return _call("POST", "/expliquer", json=payload)
