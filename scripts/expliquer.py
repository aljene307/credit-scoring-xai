"""Calcul hors ligne des explications globales (choix X04), après scripts/train.py.

Usage, depuis la racine du projet :
    .venv\\Scripts\\python.exe scripts\\expliquer.py

Écrit dans models/ (ignoré par Git) :
    shap_xgboost.parquet, shap_logreg.parquet   valeurs SHAP de 1 000 clients du test
    permutation.parquet                         importance par permutation des 4 modèles
    xgboost_sans_sensibles.joblib               XGBoost sans SEX, AGE, MARRIAGE, EDUCATION
    explications.json                           résumé : valeurs de base, équité, durées, versions
Durée sur le PC de développement : environ 5 minutes.
"""

import warnings

from credit_scoring.config import CATEGORICAL_FEATURES, MODELS_DIR
from credit_scoring.global_explanations import compute_and_save
from credit_scoring.reference import load_reference, reference_split


def main() -> None:
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    results, models = load_reference()
    X_train, X_test, y_train, y_test = reference_split(results["meta"])
    compute_and_save(MODELS_DIR, models, results, X_train, X_test, y_train, y_test, CATEGORICAL_FEATURES)
    print(f"Explications enregistrées dans {MODELS_DIR.name}/")


if __name__ == "__main__":
    main()
