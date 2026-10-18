"""Module 5 : explicabilité globale et locale (SHAP, LIME), analyse d'équité (choix X01 à X10)."""

import numpy as np
import pandas as pd
import streamlit as st

from credit_scoring import api_client, config
from credit_scoring.config import CATEGORICAL_FEATURES, LABELS, PAY_STATUS_FEATURES, PAY_STATUS_LABELS
from credit_scoring.explain import (
    IMPORTANCE_LABELS,
    LIME_FEATURES,
    LIME_SAMPLES,
    LIME_SEED,
    LimeExplainer,
    ShapExplainer,
    agreement,
    model_importance,
    summary_sentence,
    top_features,
)
from credit_scoring.global_explanations import load_global, reference_background
from credit_scoring.models import MODEL_LABELS
from credit_scoring.plots import (
    beeswarm_figure,
    dependence_figure,
    importance_figure,
    lime_figure,
    permutation_figure,
    waterfall_figure,
)
from credit_scoring.reference import load_reference, reference_split

GROUP_NAMES = {"sexe": "le sexe", "âge": "l'âge"}
SHAP_CHOICES = {"xgboost": "XGBoost (référence)", "logreg": "Régression logistique"}
LOG_ODDS_NOTE = (
    "Pour la régression logistique, les valeurs sont en **log-odds** : le logarithme du rapport entre la probabilité "
    "de défaut et celle de non-défaut ; une contribution de +0,69 double ce rapport, une contribution négative le réduit."
)


@st.cache_resource(show_spinner="Chargement des modèles et des données…")
def resources(path: str, modified: float):
    results, models = load_reference()
    X_train, X_test, y_train, y_test = reference_split(results["meta"])
    background = reference_background(models["xgboost"], X_train)
    explainers = {name: ShapExplainer(models[name], background, CATEGORICAL_FEATURES) for name in SHAP_CHOICES}
    background_proba = {name: float(models[name].predict_proba(background)[:, 1].mean()) for name in SHAP_CHOICES}
    return results, models, X_train, X_test, y_test, explainers, background_proba, LimeExplainer(X_train, CATEGORICAL_FEATURES)


@st.cache_data(show_spinner="Lecture des explications globales…")
def global_explanations(path: str, modified: float):
    return load_global(config.MODELS_DIR)


@st.cache_data(show_spinner="Explication LIME (10 000 voisins)…")
def lime_explanation(_lime, _model, client: pd.Series, key: str):
    return _lime.explain(_model, client, num_samples=LIME_SAMPLES, seed=LIME_SEED)


def client_form(client: pd.Series, client_id: int) -> pd.Series:
    """Formulaire prérempli avec les valeurs d'un client ; renvoie le client, modifié s'il a été soumis.

    Le client modifié est gardé dans la session (pour ce client) : il reste affiché quand on
    change d'onglet ou de modèle.
    """
    saved = st.session_state.get("client_modifie")
    if saved is not None and saved[0] == client_id:
        client = saved[1]
    values = {}
    with st.form("client_fictif"):
        columns = st.columns(4)
        for i, feature in enumerate(client.index):
            col = columns[i % 4]
            current = int(client[feature])
            if feature in LABELS:
                options = list(LABELS[feature])
                values[feature] = col.selectbox(feature, options, index=options.index(current),
                                                format_func=lambda v, f=feature: LABELS[f][v])
            elif feature in PAY_STATUS_FEATURES:
                options = list(range(-2, 9))
                values[feature] = col.selectbox(feature, options, index=options.index(current),
                                                format_func=lambda v: f"{v} : {PAY_STATUS_LABELS[v]}")
            else:
                values[feature] = col.number_input(feature, value=current, step=1 if feature == "AGE" else 1000)
        submitted = st.form_submit_button("Expliquer ce client modifié")
    if submitted:
        st.session_state["client_modifie"] = (client_id, pd.Series(values, name=client_id))
        return st.session_state["client_modifie"][1]
    return client


st.title("5. Explicabilité")
st.markdown(
    "Pourquoi le modèle prédit-il ce qu'il prédit ? **Explications globales** (le modèle dans son ensemble) et "
    "**explications locales** (un client donné), avec SHAP et LIME. Les valeurs SHAP de XGBoost sont en **points de "
    "probabilité** : moyenne + somme des contributions = probabilité prédite du client."
)

results_path = config.MODELS_DIR / "resultats.json"
summary_path = config.MODELS_DIR / "explications.json"
if not results_path.exists():
    st.warning("Aucun modèle de référence : lancez d'abord `python scripts/train.py`.")
    st.stop()
results, models, X_train, X_test, y_test, explainers, background_proba, lime = resources(
    str(results_path), results_path.stat().st_mtime
)
glob = global_explanations(str(summary_path), summary_path.stat().st_mtime) if summary_path.exists() else None
if glob is None:
    st.info("Explications globales absentes : lancez `python scripts/expliquer.py` (environ 5 minutes). "
            "L'explication d'un client reste disponible.")

choice = st.radio("Modèle expliqué par SHAP et LIME", list(SHAP_CHOICES), format_func=SHAP_CHOICES.get, horizontal=True)
scale = explainers[choice].scale
if scale == "log-odds":
    st.info(LOG_ODDS_NOTE)

tabs = st.tabs(["Importance globale", "SHAP global", "Dépendance", "Explication d'un client", "Équité"])

# --- Importance globale -----------------------------------------------------------------------
with tabs[0]:
    name = st.selectbox("Importance propre au modèle", list(MODEL_LABELS), format_func=MODEL_LABELS.get, index=3)
    kind = IMPORTANCE_LABELS[name]
    st.plotly_chart(importance_figure(model_importance(models[name], list(X_test.columns), CATEGORICAL_FEATURES),
                                      MODEL_LABELS[name], kind))
    st.caption(f"Mesure : {kind}. Propre à chaque type de modèle : rapide, mais non comparable d'un modèle à l'autre.")
    if name == "logreg":
        st.caption(
            "Pour une variable catégorielle, l'écart entre modalités ne tient pas compte du nombre de clients "
            "concernés : EDUCATION est en tête à cause de la modalité « Autres ou inconnu » (1,6 % des clients). "
            "L'importance par permutation ci-dessous mesure l'effet réel sur l'ensemble des clients."
        )
    if glob is not None:
        st.plotly_chart(permutation_figure(glob["permutation"], MODEL_LABELS))
        st.caption(
            "Importance par permutation : on mélange une variable dans le jeu de test et on mesure la baisse d'AUC. "
            "Même mesure pour les quatre modèles, donc comparable. Les variables dont les barres se chevauchent ont "
            "une importance équivalente."
        )

# --- SHAP global ----------------------------------------------------------------------------------
with tabs[1]:
    if glob is None:
        st.info("Lancez `python scripts/expliquer.py` pour afficher ce graphique.")
    else:
        values = glob["valeurs_shap"][choice]
        st.plotly_chart(beeswarm_figure(values, X_test, scale))
        st.caption(
            "Une ligne par variable (de la plus influente à la moins influente), un point par client. À droite de 0, "
            "la variable augmente le risque ; la couleur donne la valeur de la variable (bleu = faible, rouge = élevée)."
        )
        factor = 100 if scale == "probabilité" else 1
        mean_abs = values.abs().mean() * factor
        st.plotly_chart(importance_figure(mean_abs, "Contribution moyenne en valeur absolue",
                                          "points de probabilité" if scale == "probabilité" else "log-odds"))

# --- Dépendance -----------------------------------------------------------------------------------
with tabs[2]:
    if glob is None:
        st.info("Lancez `python scripts/expliquer.py` pour afficher ce graphique.")
    else:
        values = glob["valeurs_shap"][choice]
        ordered = list(values.abs().mean().sort_values(ascending=False).index)
        feature = st.selectbox("Variable", ordered, index=ordered.index("PAY_0"))
        st.plotly_chart(dependence_figure(values, X_test, feature, scale))
        st.caption(
            "Chaque point est un client : valeur de la variable en abscisse, sa contribution au risque en ordonnée. "
            "La courbe orange donne la moyenne par valeur. Pour PAY_0, XGBoost montre un effet plat de -2 à 0 puis un "
            "saut à 1 et 2 mois de retard ; la régression logistique, une droite (même effet à chaque cran)."
        )

# --- Explication d'un client -----------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def api_available(url: str) -> str | None:
    """None si l'API répond, sinon la raison. Vérifié au plus toutes les 30 secondes."""
    try:
        api_client.sante()
        return None
    except api_client.ApiIndisponible as exc:
        return str(exc)


def explanation(client: pd.Series, choice: str, model, threshold: float):
    """Explication d'un client : par l'API pour XGBoost (modèle complet), sinon calcul local.

    Affiche toujours la source du résultat (choix API03).
    """
    if choice == "xgboost":
        reason = api_available(api_client.api_url())
        if reason is None:
            try:
                data, ms = api_client.expliquer({k: int(v) for k, v in client.items()}, "complet",
                                                nombre_variables=LIME_FEATURES)
                st.info(f"Résultat calculé par l'**API** ({api_client.api_url()}, modèle complet) en {ms:.0f} ms.")
                contributions = pd.Series(data["contributions_shap"])[list(client.index)]
                weights = pd.Series(data["poids_lime"])
                rules = pd.Series({r["regle"]: r["poids"] for r in data["regles_lime"]})
                return data["probabilite"], data["valeur_de_base"], contributions, weights, rules, data["phrase"]
            except api_client.ApiIndisponible as exc:
                reason = str(exc)
        st.warning(f"**Calcul local** : l'API ne répond pas ({reason}). Pour l'utiliser, lancez-la (voir la page 6).")
    else:
        st.caption("**Calcul local** : l'API sert uniquement les modèles XGBoost.")
    row = client.to_frame().T.astype(X_test.dtypes.to_dict())
    proba = float(model.predict_proba(row)[:, 1][0])
    values, base = explainers[choice].explain(row)
    contributions = values.iloc[0]
    weights, rules = lime_explanation(lime, model, client, f"{choice}-{client.to_dict()}")
    sentence = summary_sentence(contributions, client, proba, background_proba[choice], threshold,
                                float(X_train["LIMIT_BAL"].median()))
    return proba, base, contributions, weights, rules, sentence


def explain_client(client_id: int) -> None:
    real = X_test.loc[client_id]
    with st.expander("Modifier ce client (client fictif)"):
        client = client_form(real, client_id)
        if st.button("Revenir au client réel"):
            st.session_state.pop("client_modifie", None)
            client = real
    fictive = not client.equals(real)
    model = models[choice]
    threshold = results["modeles"][choice]["seuils"]["f1_max"]["seuil"]
    proba, base, contributions, weights, rules, sentence = explanation(client, choice, model, threshold)

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilité de défaut", f"{proba:.1%}")
    c2.metric("Seuil recommandé", f"{threshold:.1%}")
    c3.metric("Décision", "Signalé comme risqué" if proba >= threshold else "Non signalé")
    if fictive:
        st.caption("Client fictif : valeurs modifiées dans le formulaire.")
    else:
        outcome = "a fait défaut" if y_test.loc[client_id] == 1 else "n'a pas fait défaut"
        st.caption(f"Client réel du jeu de test : il {outcome} le mois suivant.")

    st.success(sentence)

    col_shap, col_lime = st.columns(2)
    col_shap.plotly_chart(waterfall_figure(contributions, client, base, scale))
    total = base + contributions.sum()
    check = f"{base * 100:.2f} + {contributions.sum() * 100:.2f} = {total * 100:.2f} %" if scale == "probabilité" \
        else f"{base:.3f} + {contributions.sum():.3f} = {total:.3f} log-odds (soit {proba:.1%})"
    col_shap.caption(f"Valeur de base + somme des contributions : {check}.")
    col_lime.plotly_chart(lime_figure(rules))
    col_lime.caption(
        "LIME fabrique 10 000 clients voisins, observe les prédictions du modèle et ajuste un modèle linéaire "
        "simple autour du client. Graine fixe : explication reproductible."
    )

    st.subheader("SHAP et LIME sont-ils d'accord ?")
    result = agreement(contributions, weights)
    shap_top, lime_top = top_features(contributions, 5), top_features(weights, 5)
    comparison = pd.DataFrame({
        "rang": range(1, 6),
        "SHAP": [f"{f} ({'+' if contributions[f] > 0 else '−'})" for f in shap_top],
        "LIME": [f"{f} ({'+' if weights[f] > 0 else '−'})" for f in lime_top],
    })
    st.dataframe(comparison, hide_index=True)
    st.markdown(f"**{result['communes']} variables communes sur 5**, dont {result['meme_signe']} de même sens. "
                "Sur 20 clients du test, l'accord moyen mesuré est de 3,8 sur 5 pour XGBoost.")


with tabs[3]:
    st.session_state.setdefault("client_id", 1358)
    col_id, col_random = st.columns([3, 1])
    if col_random.button("Client au hasard"):
        st.session_state["client_id"] = int(np.random.default_rng().choice(X_test.index))
    client_id = int(col_id.number_input("Identifiant d'un client du jeu de test", min_value=int(X_test.index.min()),
                                        max_value=int(X_test.index.max()), step=1, key="client_id"))
    if client_id in X_test.index:
        explain_client(client_id)
    else:
        st.warning("Cet identifiant n'est pas dans le jeu de test (il fait partie de l'entraînement).")

# --- Équité ------------------------------------------------------------------------------------------
with tabs[4]:
    if glob is None:
        st.info("Lancez `python scripts/expliquer.py` pour afficher l'analyse d'équité.")
    else:
        fair = glob["equite"]
        st.markdown(f"Variables sensibles : **{', '.join(fair['variables_sensibles'])}**.")
        a, w = fair["auc_validation_avec"], fair["auc_validation_sans"]
        c1, c2 = st.columns(2)
        c1.metric("AUC de validation avec ces variables", f"{a['moyenne']:.4f}")
        c2.metric("AUC de validation sans ces variables", f"{w['moyenne']:.4f}", f"{w['moyenne'] - a['moyenne']:+.4f}")
        st.markdown("**Peut-on retrouver chaque variable sensible à partir des autres ?** (variables relais)")
        st.dataframe(pd.DataFrame(fair["relais"]).round(3), hide_index=True)
        st.caption("AUC de 0,5 ou R² proche de 0 : les autres variables ne permettent pas de retrouver la variable sensible.")
        variant = st.radio("Modèle", ["avec", "sans"], horizontal=True,
                           format_func=lambda v: f"XGBoost {v} variables sensibles")
        for group, by_variant in fair["groupes"].items():
            data = by_variant[variant]
            st.markdown(f"**Erreurs selon {GROUP_NAMES.get(group, group)}** (test, seuil recommandé)")
            st.dataframe(pd.DataFrame(data["tableau"]).round(1), hide_index=True)
            tests = data["tests"]
            st.caption(
                f"Test du khi-deux : rappel p = {tests['rappel']:.4f}, faux positifs p = {tests['faux positifs']:.4f} "
                "(p < 0,05 : l'écart entre groupes n'est probablement pas dû au hasard)."
            )
        st.info(
            "Les groupes les plus souvent signalés sont aussi ceux dont le taux de défaut réel est le plus élevé, et le "
            "modèle est calibré dans chaque groupe. Avec des taux de défaut différents, un modèle calibré ne peut pas "
            "avoir le même taux de faux positifs dans chaque groupe : le critère d'équité visé est un choix du décideur. "
            "Recommandation : déployer XGBoost sans variables sensibles (AUC −0,002), en gardant le modèle complet ici "
            "pour l'analyse."
        )
