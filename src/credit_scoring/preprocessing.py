"""Séparation entraînement / test et construction du prétraitement (module 2)."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_scoring.config import RANDOM_STATE, TEST_SIZE


def split_feature_types(
    df: pd.DataFrame, features: list[str], categorical: list[str]
) -> tuple[list[str], list[str]]:
    """Répartit les variables explicatives en numériques et catégorielles.

    Les variables déclarées catégorielles le restent (choix P01) ; une variable
    non numérique non déclarée (texte d'un CSV importé) est ajoutée aux
    catégorielles, car elle ne peut pas être imputée par une médiane.
    """
    cat = [c for c in features if c in categorical or not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in features if c not in cat]
    return num, cat


def split_data(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Séparation stratifiée (choix P04) : X_train, X_test, y_train, y_test."""
    return train_test_split(
        df[features],
        df[target],
        test_size=test_size,
        stratify=df[target],
        random_state=random_state,
    )


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Prétraitement à placer en tête du Pipeline de chaque modèle.

    scale=True (régression logistique) : imputation puis standardisation des
    variables numériques. scale=False (arbres) : imputation seulement, pour garder
    les unités d'origine dans les règles et les graphiques SHAP (choix P05).
    Les catégorielles sont toujours imputées puis encodées en one-hot (choix P06).
    La sortie est un DataFrame aux noms de colonnes lisibles (ex. EDUCATION_4).
    """
    numeric_steps = [("imputation", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("standardisation", StandardScaler()))
    categorical_steps = [
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    preprocessor = ColumnTransformer(
        [
            ("numeriques", Pipeline(numeric_steps), numeric),
            ("categorielles", Pipeline(categorical_steps), categorical),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")
