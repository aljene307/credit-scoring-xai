"""Module 3 : modèles de référence, tableau comparatif, nouveau GridSearch depuis l'interface."""

import pandas as pd
import streamlit as st

from credit_scoring import config
from credit_scoring.comparison import (
    REFERENCE_MODEL,
    comparison_table,
    default_selection,
    estimate_seconds,
    thresholds_table,
)
from credit_scoring.models import MODEL_LABELS, default_grid, n_combinations
from credit_scoring.plots import auc_comparison_figure
from credit_scoring.training import CV_FOLDS, load_results, train_model

ROUNDING = {
    "AUC": 4, "écart-type AUC": 4, "Gini": 4, "PR-AUC": 4, "AUC entraînement": 4,
    "écart entr. − valid.": 4, "proba. moyenne (%)": 2, "taux réel (%)": 2, "durée (s)": 1, "taille (Mo)": 3,
}


@st.cache_data
def load_reference(path: str, modified: float) -> dict:
    """Résultats de référence ; `modified` invalide le cache après un nouvel entraînement."""
    return load_results(path)


@st.cache_data
def load_grid_table(path: str, modified: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def show_value(value) -> str:
    return "sans limite" if value is None else str(value)


def show_comparison(summaries: dict) -> None:
    table = comparison_table(summaries)
    st.plotly_chart(auc_comparison_figure(table, MODEL_LABELS.get(REFERENCE_MODEL)))
    st.dataframe(table.drop(columns="_cle").round(ROUNDING), hide_index=True)
    st.markdown("**Seuils de décision** (choisis sur les prédictions hors pli de l'entraînement, jamais sur le test)")
    st.dataframe(thresholds_table(summaries).round(3), hide_index=True)


def preparation_matches(meta: dict, prep: dict | None) -> bool:
    """Vrai si la préparation de la page 2 est celle des modèles de référence."""
    if prep is None:
        return True
    return (
        prep["target"] == meta["cible"]
        and list(prep["X_train"].columns) == meta["variables"]
        and prep["test_size"] == meta["part_test"]
        and prep["seed"] == meta["graine"]
    )


st.title("3. Modélisation")
st.markdown(
    "Quatre modèles : deux basiques (régression logistique, arbre de décision) et deux ensemblistes "
    "(Random Forest, XGBoost). Hyperparamètres réglés par **GridSearchCV** : validation croisée à "
    f"{CV_FOLDS} plis sur le jeu d'entraînement seulement, prétraitement dans le Pipeline, choix sur l'AUC."
)
prep = st.session_state.get("preparation")

# --- Modèles de référence ---------------------------------------------------------
st.header("Modèles de référence")
results_path = config.MODELS_DIR / "resultats.json"
reference = None
if not results_path.exists():
    st.warning(
        "Aucun modèle de référence trouvé. Ils s'entraînent hors ligne avec "
        "`python scripts/train.py` (environ 10 minutes) ; vous pouvez aussi lancer un GridSearch plus bas."
    )
else:
    results = load_reference(str(results_path), results_path.stat().st_mtime)
    meta, reference = results["meta"], results["modeles"]
    versions = meta["versions"]
    st.caption(
        f"Entraînés hors ligne sur {meta['lignes_entrainement']:,} lignes ".replace(",", " ")
        + f"(jeu Taiwan, {len(meta['variables'])} variables, test {meta['part_test']:.0%}, graine {meta['graine']}) "
        f"· scikit-learn {versions['scikit-learn']} · XGBoost {versions['xgboost']}"
    )
    if not preparation_matches(meta, prep):
        st.info(
            "La préparation choisie en page 2 diffère de celle des modèles de référence : "
            "relancez un GridSearch plus bas pour obtenir des modèles adaptés."
        )
    if REFERENCE_MODEL in reference:
        st.success(
            f"**Modèle de référence : {MODEL_LABELS[REFERENCE_MODEL]}**, choisi sur la validation croisée, "
            "avant toute utilisation du jeu de test. Meilleure AUC (à égalité avec Random Forest), fichier "
            "30 fois plus léger, surapprentissage plus faible, probabilités cohérentes avec le taux réel."
        )
    show_comparison(reference)

    st.subheader("Détail par modèle")
    tabs = st.tabs([reference[name]["libelle"] for name in reference])
    for tab, name in zip(tabs, reference):
        summary = reference[name]
        with tab:
            params = pd.DataFrame(
                {
                    "valeur retenue": [show_value(v) for v in summary["meilleurs_hyperparametres"].values()],
                    "valeurs essayées": [
                        ", ".join(show_value(v) for v in default_grid(name).get(k, []))
                        for k in summary["meilleurs_hyperparametres"]
                    ],
                },
                index=list(summary["meilleurs_hyperparametres"]),
            )
            st.dataframe(params)
            if summary["bord_de_grille"]:
                st.caption(
                    "En bord de grille : " + ", ".join(summary["bord_de_grille"])
                    + ". Écarts d'AUC avec les valeurs voisines inférieurs à la variation entre plis "
                    "(voir docs/CHOIX.md, M06)."
                )
            grid_path = config.MODELS_DIR / f"grille_{name}.parquet"
            if grid_path.exists():
                with st.expander(f"Les {summary['combinaisons']} combinaisons essayées, de la meilleure à la moins bonne"):
                    st.dataframe(load_grid_table(str(grid_path), grid_path.stat().st_mtime).round(4))

# --- Nouveau GridSearch ---------------------------------------------------------------
st.header("Relancer un GridSearch")
if prep is None:
    st.info("Préparez d'abord les données dans la page « 2. Préparation » (menu de gauche).")
    st.stop()

X_train, y_train = prep["X_train"], prep["y_train"]
st.caption(
    f"Sur la préparation de la page 2 : {len(X_train):,} lignes d'entraînement, ".replace(",", " ")
    + f"{X_train.shape[1]} variables. Les résultats restent dans la session : ils ne remplacent pas "
    "les modèles de référence."
)
names = st.multiselect(
    "Modèles à entraîner", list(MODEL_LABELS), default=["logreg"], format_func=MODEL_LABELS.get
)
grids, total_seconds, complete = {}, 0.0, True
for name in names:
    with st.expander(f"Grille : {MODEL_LABELS[name]}", expanded=True):
        selection = default_selection(name, reference)
        grid = {}
        for key, values in default_grid(name).items():
            chosen = st.multiselect(
                key,
                options=list(range(len(values))),
                default=[values.index(v) for v in selection[key]],
                format_func=lambda i, values=values: show_value(values[i]),
                key=f"grille_{name}_{key}",
            )
            grid[key] = [values[i] for i in chosen]
        if any(not v for v in grid.values()):
            st.warning("Choisissez au moins une valeur par hyperparamètre.")
            complete = False
            continue
        combos = n_combinations(grid)
        seconds = estimate_seconds(name, grid, len(X_train), reference)
        total_seconds += seconds or 0
        estimate = f"durée estimée : {seconds:.0f} s" if seconds is not None else "durée non estimée"
        st.caption(f"{combos} combinaisons × {CV_FOLDS} plis = {combos * CV_FOLDS} entraînements · {estimate}")
        grids[name] = grid

if names and complete:
    st.markdown(f"**Durée totale estimée : {total_seconds / 60:.1f} min** (sur le PC de développement)")
    if st.button("Lancer le GridSearch", type="primary"):
        trained = {}
        for name, grid in grids.items():
            with st.spinner(f"{MODEL_LABELS[name]} : GridSearch en cours…"):
                search, summary = train_model(
                    name, X_train, y_train, prep["numeric"], prep["categorical"], grid=grid
                )
            trained[name] = {"modele": search.best_estimator_, "resume": summary}
        st.session_state["modeles_session"] = trained

if st.session_state.get("modeles_session"):
    st.subheader("Résultats de ce GridSearch")
    session = st.session_state["modeles_session"]
    show_comparison({name: entry["resume"] for name, entry in session.items()})
    for name, entry in session.items():
        st.caption(f"{MODEL_LABELS[name]} : {entry['resume']['meilleurs_hyperparametres']}")
