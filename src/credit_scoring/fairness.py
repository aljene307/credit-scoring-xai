"""Analyse d'équité : variables relais et erreurs du modèle par groupe (choix X09)."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score

from credit_scoring.config import LABELS, RANDOM_STATE

AGE_BINS = [0, 25, 35, 45, 55, 200]
AGE_LABELS = ["21-25 ans", "26-35 ans", "36-45 ans", "46-55 ans", "56 ans et +"]


def age_bands(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BINS, labels=AGE_LABELS).astype(str)


def sensitive_groups(X: pd.DataFrame) -> dict[str, pd.Series]:
    """Groupes comparés : sexe et tranche d'âge (si les colonnes existent)."""
    groups = {}
    if "SEX" in X:
        groups["sexe"] = X["SEX"].map(LABELS["SEX"]).fillna(X["SEX"].astype(str))
    if "AGE" in X:
        groups["âge"] = age_bands(X["AGE"])
    return groups


def proxy_scores(X: pd.DataFrame, sensitive: list[str], cv) -> pd.DataFrame:
    """Peut-on retrouver chaque variable sensible à partir des autres variables ?

    Catégorielles : part de bonnes réponses (comparée à « toujours la modalité la plus
    fréquente ») et AUC ; AGE : R² et erreur moyenne en années.
    """
    others = [c for c in X.columns if c not in sensitive]
    rows = []
    for var in sensitive:
        if var not in X:
            continue
        y = X[var]
        if var == "AGE":
            model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
            r2 = cross_val_score(model, X[others], y, cv=5, scoring="r2").mean()
            mae = -cross_val_score(model, X[others], y, cv=5, scoring="neg_mean_absolute_error").mean()
            baseline = float((y - y.mean()).abs().mean())
            rows.append({"variable": var, "mesure": "R²", "score": r2,
                         "repère": f"erreur moyenne {mae:.1f} ans (sans modèle : {baseline:.1f} ans)"})
            continue
        model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        scoring = "roc_auc" if y.nunique() == 2 else "roc_auc_ovr"
        auc = cross_val_score(model, X[others], y, cv=cv, scoring=scoring).mean()
        accuracy = float((cross_val_predict(model, X[others], y, cv=cv) == y).mean())
        majority = float(y.value_counts(normalize=True).max())
        rows.append({"variable": var, "mesure": "AUC", "score": auc,
                     "repère": f"bonnes réponses {accuracy:.1%} (modalité la plus fréquente : {majority:.1%})"})
    return pd.DataFrame(rows)


def group_errors(y_true, proba, threshold: float, groups: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Taux de défaut, rappel et faux positifs par groupe, et tests du khi-deux.

    Le rappel est mesuré parmi les défauts réels, le taux de faux positifs parmi les
    bons payeurs ; le test dit si l'écart entre groupes peut être dû au hasard.
    """
    y = np.asarray(y_true)
    predicted = (np.asarray(proba) >= threshold).astype(int)
    g = np.asarray(groups)
    rows = []
    for value in sorted(pd.unique(g)):
        m = g == value
        defaults, goods = m & (y == 1), m & (y == 0)
        rows.append(
            {
                "groupe": value,
                "clients": int(m.sum()),
                "taux de défaut (%)": y[m].mean() * 100,
                "proba. moyenne (%)": np.asarray(proba)[m].mean() * 100,
                "clients signalés (%)": predicted[m].mean() * 100,
                "rappel (%)": predicted[defaults].mean() * 100 if defaults.any() else np.nan,
                "faux positifs (%)": predicted[goods].mean() * 100 if goods.any() else np.nan,
            }
        )
    tests = {}
    for name, mask in (("rappel", y == 1), ("faux positifs", y == 0)):
        table = pd.crosstab(g[mask], predicted[mask])
        tests[name] = float(chi2_contingency(table)[1]) if table.shape == (len(rows), 2) else float("nan")
    return pd.DataFrame(rows), tests
