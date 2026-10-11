"""Constantes du projet : chemins, graine aléatoire, noms de colonnes."""

import os
from pathlib import Path

# Racine du dépôt (src/credit_scoring/config.py -> deux niveaux au-dessus de src).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dossier du cache des données, modifiable par la variable d'environnement
# CREDIT_DATA_DIR (utile si le paquet n'est pas installé en mode modifiable).
DATA_DIR = Path(os.environ.get("CREDIT_DATA_DIR", PROJECT_ROOT / "data"))
CACHE_FILE = DATA_DIR / "credit_default.parquet"

UCI_DATASET_ID = 350
RANDOM_STATE = 42

ID_COL = "ID"
TARGET = "DEFAULT"

# Correspondance X1…X23 -> noms lisibles, vérifiée sur les métadonnées UCI
# (colonne « description » de fetch_ucirepo(id=350).variables).
FEATURE_NAMES = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
}
TARGET_SOURCE = "Y"
COLUMN_NAMES = {**FEATURE_NAMES, TARGET_SOURCE: TARGET}

# Variables codées en entiers mais sans ordre : encodées en one-hot (choix P01).
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
# Variables sensibles : gardées par défaut, exclusion possible (choix P08).
SENSITIVE_FEATURES = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]
TEST_SIZE = 0.2
# Statuts de remboursement, gardés numériques car ordinaux (choix P02).
PAY_STATUS_FEATURES = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_FEATURES = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_FEATURES = [f"PAY_AMT{i}" for i in range(1, 7)]

# Recodage des modalités non documentées (choix R1 et R2).
RECODING = {
    "EDUCATION": {0: 4, 5: 4, 6: 4},
    "MARRIAGE": {0: 3},
}

# Libellés affichés dans l'application (codes officiels UCI, après recodage).
LABELS = {
    "SEX": {1: "Homme", 2: "Femme"},
    "EDUCATION": {1: "3e cycle", 2: "Université", 3: "Lycée", 4: "Autres ou inconnu"},
    "MARRIAGE": {1: "Marié", 2: "Célibataire", 3: "Autres"},
}
PAY_STATUS_LABELS = {
    -2: "Pas d'utilisation",
    -1: "Payé en totalité",
    0: "Crédit renouvelable (minimum payé)",
    **{k: f"Retard de {k} mois" for k in range(1, 10)},
}
