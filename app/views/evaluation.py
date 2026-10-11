"""Module 4 : évaluation des modèles sur le jeu de test (choix E01 à E06)."""

import streamlit as st

from credit_scoring import config
from credit_scoring.comparison import REFERENCE_MODEL, auc_differences, holdout_table
from credit_scoring.evaluation import (
    bootstrap_auc,
    calibration_table,
    confusion_counts,
    decision_sentence,
    holdout_report,
    metrics_at_threshold,
)
from credit_scoring.models import MODEL_LABELS
from credit_scoring.plots import (
    calibration_figure,
    confusion_figure,
    ks_figure,
    precision_recall_figure,
    roc_figure,
)
from credit_scoring.reference import load_reference, reference_split

CLASS_NAMES = {0: "Pas de défaut", 1: "Défaut"}
ROUNDING = {
    "AUC validation croisée": 4, "AUC test": 4, "IC 95 % bas": 4, "IC 95 % haut": 4, "Gini": 4, "KS": 4,
    "PR-AUC": 4, "Brier": 4, "proba. moyenne (%)": 2, "seuil": 3, "précision (%)": 1, "rappel (%)": 1,
    "F1": 4, "clients signalés (%)": 1,
}


@st.cache_resource(show_spinner="Chargement des modèles de référence…")
def reference_models(path: str, modified: float):
    results, models = load_reference()
    _, X_test, _, y_test = reference_split(results["meta"])
    return results, models, X_test, y_test


@st.cache_data(show_spinner="Calcul des intervalles de confiance (1 000 tirages)…")
def bootstrap(_y_test, _probas: dict, key: str) -> dict:
    """`key` identifie les modèles évalués (les arguments préfixés par _ ne sont pas hachés)."""
    return bootstrap_auc(_y_test, _probas, n_resamples=1000)


st.title("4. Évaluation")
st.markdown(
    "Mesure des modèles sur le **jeu de test** (clients jamais vus pendant l'entraînement ni le réglage). "
    "Chaque modèle est évalué à **son seuil F1 fixé sur l'entraînement** : le test sert à mesurer, jamais à choisir."
)

# --- Choix des modèles évalués -------------------------------------------------
results_path = config.MODELS_DIR / "resultats.json"
session = st.session_state.get("modeles_session")
sources = []
if results_path.exists():
    sources.append("Modèles de référence")
if session:
    sources.append("Modèles du GridSearch de la page 3")
if not sources:
    st.warning(
        "Aucun modèle à évaluer : entraînez les modèles de référence (`python scripts/train.py`) "
        "ou lancez un GridSearch dans la page « 3. Modélisation »."
    )
    st.stop()
source = st.radio("Modèles évalués", sources, horizontal=True)

if source == "Modèles de référence":
    results, models, X_test, y_test = reference_models(str(results_path), results_path.stat().st_mtime)
    summaries = results["modeles"]
    key = f"reference-{results_path.stat().st_mtime}"
else:
    prep = st.session_state["preparation"]
    models = {name: entry["modele"] for name, entry in session.items()}
    summaries = {name: entry["resume"] for name, entry in session.items()}
    X_test, y_test = prep["X_test"], prep["y_test"]
    key = "session-" + "-".join(f"{n}{s['meilleurs_hyperparametres']}" for n, s in summaries.items())

probas = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}
labels = {name: MODEL_LABELS[name] for name in models}
thresholds = {name: summaries[name]["seuils"]["f1_max"]["seuil"] for name in models}
reference = REFERENCE_MODEL if REFERENCE_MODEL in models else next(iter(models))
st.caption(
    f"Jeu de test : {len(y_test):,} clients, dont {int(y_test.sum()):,} défauts ".replace(",", " ")
    + f"({y_test.mean():.2%})."
)

# --- Tableau des métriques ---------------------------------------------------------
st.header("Métriques sur le test")
boot = bootstrap(y_test, probas, key)
table = holdout_table(y_test, probas, summaries, boot)
st.dataframe(table.drop(columns="_cle").round(ROUNDING), hide_index=True)
st.caption(
    "IC 95 % : intervalle de confiance de l'AUC par bootstrap (1 000 tirages des clients du test). "
    "Gini = 2 × AUC − 1. Brier : erreur moyenne au carré des probabilités (plus bas = mieux ; "
    f"un modèle qui annoncerait {y_test.mean():.1%} à tous obtiendrait {y_test.mean() * (1 - y_test.mean()):.3f})."
)
if len(models) > 1:
    st.markdown(f"**Écarts d'AUC avec {labels[reference]}** (mêmes tirages pour tous les modèles)")
    st.dataframe(auc_differences(boot, summaries, reference).round(4), hide_index=True)

col_roc, col_pr = st.columns(2)
col_roc.plotly_chart(roc_figure(y_test, probas, labels, thresholds))
col_pr.plotly_chart(precision_recall_figure(y_test, probas, labels, thresholds))
st.caption("Les points marquent le seuil retenu de chaque modèle.")

# --- Analyse au seuil -----------------------------------------------------------------
st.header("Effet du seuil de décision")
st.info(
    "Démonstration : le curseur agit sur le jeu de test pour montrer le compromis. Le seuil recommandé, "
    "lui, a été choisi sur l'entraînement (F1 maximal sur les prédictions hors pli)."
)
names = list(models)
name = st.selectbox(
    "Modèle", names, index=names.index(reference), format_func=lambda n: labels[n], key="modele_seuil"
)
recommended = summaries[name]["seuils"]["f1_max"]["seuil"]
ks_threshold = summaries[name]["seuils"]["ks"]["seuil"]
position = st.slider(
    "Seuil de décision", 0.001, 0.999, round(recommended, 3), 0.001, format="%.3f", key=f"seuil_{name}",
    help=f"Recommandé (F1 maximal) : {recommended:.4f} · KS : {ks_threshold:.4f}",
)
# Sur la position du seuil recommandé, on applique sa valeur exacte (pas son arrondi) :
# le premier affichage correspond alors exactement au tableau des métriques.
threshold = recommended if position == round(recommended, 3) else position
st.caption(
    f"Seuil appliqué : {threshold:.4f} · recommandé (F1 maximal) : {recommended:.4f} · KS : {ks_threshold:.4f}"
)

current = metrics_at_threshold(y_test, probas[name], threshold)
at_recommended = metrics_at_threshold(y_test, probas[name], recommended)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Précision", f"{current['precision']:.1%}", f"{(current['precision'] - at_recommended['precision']) * 100:+.1f} pts")
c2.metric("Rappel", f"{current['rappel']:.1%}", f"{(current['rappel'] - at_recommended['rappel']) * 100:+.1f} pts")
c3.metric("F1", f"{current['f1']:.3f}", f"{current['f1'] - at_recommended['f1']:+.3f}")
c4.metric("Clients signalés", f"{current['part_signales']:.1%}",
          f"{(current['part_signales'] - at_recommended['part_signales']) * 100:+.1f} pts", delta_color="off")
st.caption("Variations par rapport au seuil recommandé.")

counts = confusion_counts(y_test, probas[name], threshold)
st.plotly_chart(confusion_figure(counts))
st.success(decision_sentence(counts))

# --- KS et calibration ------------------------------------------------------------------
st.header("Séparation des classes et calibration")
col_ks, col_cal = st.columns(2)
col_ks.plotly_chart(ks_figure(y_test, probas[name], CLASS_NAMES))
col_ks.caption(
    "Part cumulée des clients dont le score est inférieur à chaque valeur. Le KS est le plus grand écart "
    "entre bons payeurs et défauts : plus il est grand, mieux le score sépare les deux groupes."
)
calibration = calibration_table(y_test, probas[name])
report = holdout_report(y_test, probas[name], recommended)
col_cal.plotly_chart(calibration_figure(calibration, labels[name]))
col_cal.caption(
    f"Score de Brier : {report['brier']:.4f}. Si les points suivent la diagonale, une probabilité annoncée de "
    "30 % correspond bien à 30 % de défauts observés."
)
with st.expander("Tableau de calibration"):
    shown = calibration.assign(
        **{"écart (points)": (calibration["taux_observe"] - calibration["proba_moyenne"]) * 100}
    )
    st.dataframe(shown.round(4), hide_index=True)
