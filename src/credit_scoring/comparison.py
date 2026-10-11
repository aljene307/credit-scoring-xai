"""Tableaux comparatifs des modèles à partir des résumés d'entraînement (étape 3)."""

import pandas as pd

from credit_scoring.evaluation import confidence_interval, holdout_report
from credit_scoring.models import default_grid, n_combinations

REFERENCE_MODEL = "xgboost"  # choix M07, fait sur la validation croisée avant le test
REFERENCE_TRAIN_ROWS = 24_000


def comparison_table(summaries: dict) -> pd.DataFrame:
    """Une ligne par modèle : scores de validation croisée, surapprentissage, probabilités."""
    rows = []
    for name, s in summaries.items():
        cv = s["validation_croisee"]
        auc = cv["roc_auc"]["moyenne"]
        rows.append(
            {
                "modèle": s["libelle"],
                "AUC": auc,
                "écart-type AUC": cv["roc_auc"]["ecart_type"],
                "Gini": 2 * auc - 1,
                "PR-AUC": cv["pr_auc"]["moyenne"],
                "AUC entraînement": s["auc_entrainement"],
                "écart entr. − valid.": s["auc_entrainement"] - auc,
                "proba. moyenne (%)": s["proba_moyenne_hors_pli"] * 100,
                "taux réel (%)": s["taux_reel_entrainement"] * 100,
                "combinaisons": s["combinaisons"],
                "durée (s)": s["duree_s"],
                "taille (Mo)": s["taille_octets"] / 1e6 if "taille_octets" in s else None,
                "_cle": name,
            }
        )
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


def thresholds_table(summaries: dict) -> pd.DataFrame:
    """Seuil F1 maximal (recommandé) et seuil du KS, avec précision et rappel (choix E01)."""
    labels = {"f1_max": "F1 maximal (recommandé)", "ks": "KS"}
    rows = []
    for s in summaries.values():
        for key, label in labels.items():
            t = s["seuils"][key]
            rows.append(
                {
                    "modèle": s["libelle"],
                    "critère": label,
                    "seuil": t["seuil"],
                    "précision (%)": t["precision"] * 100,
                    "rappel (%)": t["rappel"] * 100,
                    "F1": t["f1"],
                    "clients signalés (%)": t["part_signales"] * 100,
                    "KS": t.get("ks"),
                }
            )
    return pd.DataFrame(rows)


def holdout_table(y_test, probas: dict, summaries: dict, bootstrap: dict) -> pd.DataFrame:
    """Métriques sur le test de chaque modèle, à son seuil F1 fixé sur l'entraînement (E04, E05)."""
    rows = []
    for name, summary in summaries.items():
        report = holdout_report(y_test, probas[name], summary["seuils"]["f1_max"]["seuil"])
        low, high = confidence_interval(bootstrap[name])
        rows.append(
            {
                "modèle": summary["libelle"],
                "AUC validation croisée": summary["validation_croisee"]["roc_auc"]["moyenne"],
                "AUC test": report["auc"],
                "IC 95 % bas": low,
                "IC 95 % haut": high,
                "Gini": report["gini"],
                "KS": report["ks"],
                "PR-AUC": report["pr_auc"],
                "Brier": report["brier"],
                "proba. moyenne (%)": report["proba_moyenne"] * 100,
                "seuil": report["seuil"],
                "précision (%)": report["precision"] * 100,
                "rappel (%)": report["rappel"] * 100,
                "F1": report["f1"],
                "clients signalés (%)": report["part_signales"] * 100,
                "_cle": name,
            }
        )
    return pd.DataFrame(rows).sort_values("AUC test", ascending=False).reset_index(drop=True)


def auc_differences(bootstrap: dict, summaries: dict, reference: str = REFERENCE_MODEL) -> pd.DataFrame:
    """Écart d'AUC entre la référence et chaque autre modèle, sur les mêmes tirages."""
    rows = []
    for name, summary in summaries.items():
        if name == reference or reference not in bootstrap:
            continue
        diff = bootstrap[reference] - bootstrap[name]
        low, high = confidence_interval(diff)
        rows.append(
            {
                "comparaison": f"{summaries[reference]['libelle']} − {summary['libelle']}",
                "écart d'AUC": float(diff.mean()),
                "IC 95 % bas": low,
                "IC 95 % haut": high,
                "conclusion": "égalité (l'intervalle contient 0)" if low <= 0 <= high
                else ("référence meilleure" if low > 0 else "référence moins bonne"),
            }
        )
    return pd.DataFrame(rows)


def estimate_seconds(name: str, grid: dict, n_rows: int, reference: dict | None) -> float | None:
    """Durée estimée d'un GridSearch, proportionnelle à celle du GridSearch de référence.

    Hypothèse simple : durée proportionnelle au nombre de combinaisons et au nombre
    de lignes. Sans résultat de référence, pas d'estimation.
    """
    if not reference or name not in reference:
        return None
    ref = reference[name]
    per_combination = ref["duree_s"] / ref["combinaisons"]
    return per_combination * n_combinations(grid) * n_rows / REFERENCE_TRAIN_ROWS


def default_selection(name: str, reference: dict | None) -> dict:
    """Valeurs cochées par défaut dans l'interface : les meilleures de la référence,
    sinon la première valeur de chaque hyperparamètre (une seule combinaison)."""
    grid = default_grid(name)
    best = (reference or {}).get(name, {}).get("meilleurs_hyperparametres", {})
    return {key: [best[key]] if key in best and best[key] in values else values[:1] for key, values in grid.items()}
