"""Téléchargement, renommage et cache du jeu Default of Credit Card Clients (UCI 350)."""

from pathlib import Path

import pandas as pd

from credit_scoring.config import (
    CACHE_FILE,
    COLUMN_NAMES,
    FEATURE_NAMES,
    ID_COL,
    RECODING,
    UCI_DATASET_ID,
)


class DataDownloadError(RuntimeError):
    """Le jeu de données n'a pas pu être téléchargé depuis l'UCI."""


def check_variable_names(variables: pd.DataFrame) -> None:
    """Vérifie que les descriptions UCI de X1…X23 correspondent aux noms attendus.

    Protège contre un changement de l'ordre des colonnes côté UCI : le renommage
    se ferait alors en silence sur les mauvaises variables.
    """
    described = dict(zip(variables["name"], variables["description"]))
    mismatches = {
        code: (expected, described.get(code))
        for code, expected in FEATURE_NAMES.items()
        if described.get(code) != expected
    }
    if mismatches:
        details = ", ".join(f"{code}: attendu {exp}, reçu {got}" for code, (exp, got) in mismatches.items())
        raise ValueError(f"Les métadonnées UCI ne correspondent plus au renommage prévu ({details}).")


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme X1…X23 et Y en noms lisibles ; échoue si une colonne manque."""
    missing = [col for col in COLUMN_NAMES if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes du jeu de données : {missing}")
    return df.rename(columns=COLUMN_NAMES)


def download_credit_data() -> pd.DataFrame:
    """Télécharge le jeu depuis l'UCI, vérifie les métadonnées et renomme les colonnes.

    Le résultat est indexé par l'identifiant client (ID).
    """
    from ucimlrepo import fetch_ucirepo

    try:
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    except Exception as exc:  # réseau, serveur UCI indisponible, réponse invalide
        raise DataDownloadError(
            f"Téléchargement du jeu UCI {UCI_DATASET_ID} impossible : {exc}"
        ) from exc

    check_variable_names(dataset.variables)
    raw = pd.concat([dataset.data.ids, dataset.data.features, dataset.data.targets], axis=1)
    return rename_columns(raw).set_index(ID_COL)


def load_credit_data(cache_path: Path = CACHE_FILE, refresh: bool = False) -> pd.DataFrame:
    """Charge le jeu depuis le cache Parquet, ou le télécharge et le met en cache."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    df = download_credit_data()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def recode_undocumented(df: pd.DataFrame) -> pd.DataFrame:
    """Regroupe les modalités non documentées (EDUCATION 0/5/6 -> 4, MARRIAGE 0 -> 3).

    Règle fixe appliquée ligne par ligne : elle n'apprend rien des données et
    peut donc être appliquée avant la séparation entraînement / test.
    Les PAY_* ne sont pas modifiées (choix R3).
    """
    recoded = df.copy()
    for col, mapping in RECODING.items():
        recoded[col] = recoded[col].replace(mapping)
    return recoded
