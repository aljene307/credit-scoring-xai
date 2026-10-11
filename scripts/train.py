"""Entraînement hors ligne des modèles de référence (choix M04).

Usage, depuis la racine du projet :
    .venv\\Scripts\\python.exe scripts\\train.py                       (les quatre modèles)
    .venv\\Scripts\\python.exe scripts\\train.py --modeles logreg arbre

Écrit dans models/ (ignoré par Git) :
    <modele>.joblib          meilleur Pipeline, réentraîné sur tout le jeu d'entraînement
    grille_<modele>.parquet  toutes les combinaisons essayées et leurs scores
    resultats.json           résumé : hyperparamètres, scores, seuils, tailles, versions
"""

import argparse
import platform
import time

import sklearn
import xgboost

from credit_scoring.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from credit_scoring.data import load_credit_data, recode_undocumented
from credit_scoring.models import MODEL_LABELS
from credit_scoring.preprocessing import split_data, split_feature_types
from credit_scoring.training import (
    CV_FOLDS,
    cv_results_table,
    load_results,
    save_model,
    save_results,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modeles", nargs="+", choices=list(MODEL_LABELS), default=list(MODEL_LABELS))
    args = parser.parse_args()

    df = recode_undocumented(load_credit_data())
    features = [c for c in df.columns if c != TARGET]
    numeric, categorical = split_feature_types(df, features, CATEGORICAL_FEATURES)
    X_train, X_test, y_train, _ = split_data(df, TARGET, features, TEST_SIZE, RANDOM_STATE)
    print(f"Entraînement : {len(X_train)} lignes | test mis de côté : {len(X_test)} lignes")

    results_path = MODELS_DIR / "resultats.json"
    results = load_results(results_path) if results_path.exists() else {"modeles": {}}
    results["meta"] = {
        "cible": TARGET,
        "variables": features,
        "numeriques": numeric,
        "categorielles": categorical,
        "part_test": TEST_SIZE,
        "graine": RANDOM_STATE,
        "plis": CV_FOLDS,
        "lignes_entrainement": len(X_train),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    for name in args.modeles:
        print(f"\n=== {MODEL_LABELS[name]} ===", flush=True)
        start = time.perf_counter()
        search, summary = train_model(name, X_train, y_train, numeric, categorical)
        summary["taille_octets"] = save_model(search.best_estimator_, MODELS_DIR / f"{name}.joblib")
        results["modeles"][name] = summary
        save_results(results, results_path)  # dès la fin de chaque modèle : rien n'est perdu en cas d'arrêt
        cv_results_table(search).to_parquet(MODELS_DIR / f"grille_{name}.parquet")

        cv = summary["validation_croisee"]
        print(f"{summary['combinaisons']} combinaisons, {summary['entrainements']} entraînements, "
              f"{time.perf_counter() - start:.0f} s")
        print(f"meilleurs hyperparamètres : {summary['meilleurs_hyperparametres']}")
        print(f"AUC validation {cv['roc_auc']['moyenne']:.4f} ± {cv['roc_auc']['ecart_type']:.4f} "
              f"| AUC entraînement {summary['auc_entrainement']:.4f} "
              f"| taille {summary['taille_octets'] / 1e6:.2f} Mo", flush=True)

    print(f"\nRésultats enregistrés dans {results_path.name}")


if __name__ == "__main__":
    main()
