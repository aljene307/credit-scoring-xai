"""Module 1 : chargement des données et analyse exploratoire."""

import io

import pandas as pd
import streamlit as st

from credit_scoring.config import CATEGORICAL_FEATURES, TARGET
from credit_scoring.data import (
    DataDownloadError,
    binarize_target,
    load_credit_data,
    recode_undocumented,
)
from credit_scoring.eda import class_balance, is_discrete, missing_summary, positive_rate_by
from credit_scoring.plots import (
    class_balance_figure,
    correlation_figure,
    distribution_figure,
    positive_rate_figure,
)

SOURCE_TAIWAN = "Jeu Taiwan (UCI, téléchargement automatique)"
SOURCE_CSV = "Importer un fichier CSV"


@st.cache_data(show_spinner="Chargement du jeu de données…")
def load_taiwan() -> pd.DataFrame:
    return recode_undocumented(load_credit_data())


@st.cache_data(show_spinner="Lecture du fichier…")
def read_csv(content: bytes) -> pd.DataFrame:
    # sep=None : le séparateur (virgule, point-virgule…) est détecté automatiquement.
    return pd.read_csv(io.BytesIO(content), sep=None, engine="python")


def choose_taiwan() -> dict | None:
    try:
        df = load_taiwan()
    except DataDownloadError as exc:
        st.error(f"{exc}\n\nVérifiez la connexion Internet puis rechargez la page.")
        return None
    return {
        "df": df,
        "target": TARGET,
        "categorical": CATEGORICAL_FEATURES,
        "class_names": {0: "Pas de défaut", 1: "Défaut"},
        "source": "Default of Credit Card Clients (UCI 350), modalités non documentées regroupées",
    }


def choose_csv() -> dict | None:
    uploaded = st.file_uploader("Fichier CSV", type="csv")
    if uploaded is None:
        st.info("Choisissez un fichier CSV pour continuer.")
        return None
    try:
        raw = read_csv(uploaded.getvalue())
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Le fichier n'a pas pu être lu comme un CSV : {exc}")
        return None

    col_target, col_positive = st.columns(2)
    target = col_target.selectbox("Variable cible", raw.columns, index=len(raw.columns) - 1)
    values = raw[target].dropna().unique()
    if len(values) != 2:
        st.error(
            f"La cible doit avoir exactement deux valeurs (classification binaire) ; "
            f"« {target} » en a {len(values)}."
        )
        return None
    positive = col_positive.selectbox(
        "Valeur de la classe positive (événement à prédire)", values, index=1
    )
    df = binarize_target(raw, target, positive)
    if len(df) < len(raw):
        st.warning(f"{len(raw) - len(df)} lignes sans valeur de cible ont été retirées.")

    features = [c for c in df.columns if c != target]
    suggested = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    categorical = st.multiselect(
        "Variables catégorielles (les codes numériques sans ordre doivent être ajoutés ici)",
        features,
        default=suggested,
    )
    negative = next(v for v in values if v != positive)
    return {
        "df": df,
        "target": target,
        "categorical": categorical,
        "class_names": {0: f"{target} = {negative}", 1: f"{target} = {positive}"},
        "source": f"Fichier importé : {uploaded.name}",
    }


st.title("1. Données")

source = st.radio("Source des données", [SOURCE_TAIWAN, SOURCE_CSV], horizontal=True)
dataset = choose_taiwan() if source == SOURCE_TAIWAN else choose_csv()
if dataset is None:
    st.stop()
st.session_state["dataset"] = dataset  # partagé avec les pages suivantes

df, target = dataset["df"], dataset["target"]
categorical, class_names = dataset["categorical"], dataset["class_names"]
features = [c for c in df.columns if c != target]
overall_rate = df[target].mean() * 100

st.caption(dataset["source"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes", f"{len(df):,}".replace(",", " "))
c2.metric("Variables explicatives", len(features))
c3.metric("Valeurs manquantes", int(df.isna().sum().sum()))
c4.metric(f"Taux de « {class_names[1]} »", f"{overall_rate:.1f} %")

tabs = st.tabs(
    ["Aperçu", "Statistiques", "Valeurs manquantes", "Classes", "Distributions", "Corrélations", "Taux par variable"]
)

with tabs[0]:
    n_rows = st.slider("Nombre de lignes affichées", 5, 100, 10)
    st.dataframe(df.head(n_rows))

with tabs[1]:
    st.dataframe(df.describe().T.round(2))
    if categorical:
        st.markdown("**Variables catégorielles** (nombre de modalités)")
        st.dataframe(df[categorical].nunique().rename("modalités"))

with tabs[2]:
    missing = missing_summary(df)
    if missing.empty:
        st.success("Aucune valeur manquante.")
    else:
        st.dataframe(missing)

with tabs[3]:
    balance = class_balance(df[target])
    st.plotly_chart(class_balance_figure(balance, class_names))
    st.dataframe(balance.rename(index=class_names))
    if overall_rate < 40:
        st.caption(
            "Classes déséquilibrées : le taux de bonnes réponses est trompeur. "
            "Les modèles seront comparés sur l'AUC, le rappel et la PR-AUC."
        )

with tabs[4]:
    col_var, col_norm = st.columns([3, 1])
    variable = col_var.selectbox("Variable", features, key="dist_var")
    normalize = col_norm.checkbox("En % de chaque classe", value=True)
    st.plotly_chart(
        distribution_figure(df, variable, target, variable in categorical, class_names, normalize)
    )

with tabs[5]:
    method = st.radio("Méthode", ["Pearson", "Spearman"], horizontal=True)
    numeric = [c for c in features if c not in categorical and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        st.info("Aucune variable numérique non catégorielle.")
    else:
        corr = df[numeric + [target]].corr(method=method.lower())
        st.plotly_chart(correlation_figure(corr, method))
        st.markdown(f"**Corrélation de chaque variable avec la cible ({method})**")
        with_target = corr[target].drop(target)
        st.dataframe(
            with_target.reindex(with_target.abs().sort_values(ascending=False).index)
            .round(3)
            .rename("corrélation")
        )
        st.caption(
            "Les variables catégorielles sont exclues : une corrélation sur des codes sans ordre "
            "n'a pas de sens."
        )

with tabs[6]:
    col_var, col_bins = st.columns([3, 1])
    variable = col_var.selectbox("Variable", features, key="rate_var")
    discrete = is_discrete(df[variable], variable in categorical)
    n_bins = col_bins.slider("Nombre de tranches", 4, 20, 10, disabled=discrete)
    table = positive_rate_by(df, variable, target, variable in categorical, n_bins=n_bins)
    st.plotly_chart(positive_rate_figure(table, variable, overall_rate, class_names[1]))
    st.caption(
        "Variables catégorielles ou à peu de valeurs : une barre par modalité. "
        "Variables continues : tranches de même effectif (quantiles)."
    )
    with st.expander("Tableau des valeurs"):
        st.dataframe(table)
