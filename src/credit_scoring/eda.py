"""Calculs de l'analyse exploratoire, indépendants de l'affichage."""

import pandas as pd

from credit_scoring.config import LABELS, PAY_STATUS_FEATURES, PAY_STATUS_LABELS

# En dessous de ce nombre de valeurs distinctes, une variable numérique est
# résumée valeur par valeur plutôt que par tranches (ex. PAY_* : -2 à 8).
MAX_DISCRETE_VALUES = 12


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre et part (%) de valeurs manquantes, pour les colonnes concernées."""
    counts = df.isna().sum()
    counts = counts[counts > 0]
    return pd.DataFrame(
        {"manquantes": counts, "part (%)": (counts / len(df) * 100).round(2)}
    ).sort_values("manquantes", ascending=False)


def class_balance(target: pd.Series) -> pd.DataFrame:
    """Effectif et part (%) de chaque classe de la cible."""
    counts = target.value_counts().sort_index()
    return pd.DataFrame({"clients": counts, "part (%)": (counts / counts.sum() * 100).round(2)})


def is_discrete(series: pd.Series, categorical: bool) -> bool:
    """Vrai si la variable doit être résumée valeur par valeur (et non par tranches)."""
    return (
        categorical
        or not pd.api.types.is_numeric_dtype(series)
        or series.nunique(dropna=True) <= MAX_DISCRETE_VALUES
    )


def display_value(col: str, value) -> str:
    """Libellé lisible d'une modalité (jeu Taiwan) ; la valeur brute sinon."""
    if col in LABELS and value in LABELS[col]:
        return LABELS[col][value]
    if col in PAY_STATUS_FEATURES and value in PAY_STATUS_LABELS:
        return f"{value} : {PAY_STATUS_LABELS[value]}"
    return str(value)


def positive_rate_by(
    df: pd.DataFrame, col: str, target: str, categorical: bool, n_bins: int = 10
) -> pd.DataFrame:
    """Taux de la classe positive (%) par modalité, ou par tranche de quantiles.

    Les valeurs manquantes de la variable forment leur propre groupe.
    """
    values = df[col]
    if is_discrete(values, categorical):
        groups = values.map(lambda v: "manquant" if pd.isna(v) else display_value(col, v))
        uniques = list(values.dropna().unique())
        try:
            uniques.sort()
        except TypeError:  # modalités de types mélangés (CSV importé)
            uniques.sort(key=str)
        order = [display_value(col, v) for v in uniques]
    else:
        bins = pd.qcut(values, q=n_bins, duplicates="drop")
        groups = bins.astype(str).where(bins.notna(), "manquant")
        order = [str(interval) for interval in bins.cat.categories]
    if groups.eq("manquant").any():
        order.append("manquant")

    table = df[target].groupby(groups).agg(clients="size", taux="mean")
    table["taux (%)"] = (table.pop("taux") * 100).round(1)
    table = table.reindex(order)
    table.index.name = col
    return table
