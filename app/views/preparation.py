"""Module 2 : choix des variables, séparation entraînement / test, prétraitement."""

import pandas as pd
import streamlit as st

from credit_scoring.config import RANDOM_STATE, SENSITIVE_FEATURES, TEST_SIZE
from credit_scoring.preprocessing import build_preprocessor, split_data, split_feature_types

st.title("2. Préparation")

dataset = st.session_state.get("dataset")
if dataset is None:
    st.info("Commencez par charger des données dans la page « 1. Données » (menu de gauche).")
    st.stop()

df, target = dataset["df"], dataset["target"]
class_names = dataset["class_names"]
all_features = [c for c in df.columns if c != target]
st.caption(dataset["source"])
st.markdown(f"**Variable cible :** `{target}` (1 = {class_names[1]})")

# --- Variables explicatives -------------------------------------------------
st.subheader("Variables explicatives")
sensitive = [c for c in SENSITIVE_FEATURES if c in all_features]
exclude_sensitive = False
if sensitive:
    exclude_sensitive = st.checkbox(
        f"Exclure les variables sensibles ({', '.join(sensitive)})",
        help="Utile pour l'analyse d'équité : comparer les modèles avec et sans ces variables.",
    )
available = [c for c in all_features if not (exclude_sensitive and c in sensitive)]
# La clé dépend de l'exclusion : la liste est réinitialisée quand la case change.
features = st.multiselect(
    "Variables utilisées par les modèles", available, default=available, key=f"features_{exclude_sensitive}"
)
if not features:
    st.warning("Choisissez au moins une variable explicative.")
    st.stop()
numeric, categorical = split_feature_types(df, features, dataset["categorical"])
st.caption(
    f"{len(numeric)} numériques, {len(categorical)} catégorielles"
    + (f" ({', '.join(categorical)})" if categorical else "")
)

# --- Séparation entraînement / test ------------------------------------------
st.subheader("Séparation entraînement / test")
col_size, col_seed = st.columns(2)
test_pct = col_size.slider("Part du jeu de test (%)", 10, 40, int(TEST_SIZE * 100), step=5)
seed = col_seed.number_input("Graine aléatoire", min_value=0, value=RANDOM_STATE, step=1)

X_train, X_test, y_train, y_test = split_data(df, target, features, test_pct / 100, int(seed))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Entraînement", f"{len(X_train):,}".replace(",", " "))
c2.metric("Test", f"{len(X_test):,}".replace(",", " "))
c3.metric(f"Taux « {class_names[1]} » (entraînement)", f"{y_train.mean() * 100:.2f} %")
c4.metric(f"Taux « {class_names[1]} » (test)", f"{y_test.mean() * 100:.2f} %")
st.caption(
    "Séparation stratifiée : même proportion de la classe positive dans les deux jeux. "
    "Le jeu de test est mis de côté ; il ne sert jamais au réglage des modèles."
)

st.session_state["preparation"] = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "numeric": numeric,
    "categorical": categorical,
    "target": target,
    "class_names": class_names,
    "test_size": test_pct / 100,
    "seed": int(seed),
}

# --- Prétraitement -------------------------------------------------------------
st.subheader("Prétraitement")
st.markdown(
    """
| Variables | Régression logistique | Arbre, Random Forest, XGBoost |
|---|---|---|
| Numériques | médiane si manquante, puis **standardisation** | médiane si manquante, **unités d'origine** |
| Catégorielles | modalité la plus fréquente si manquante, puis **one-hot** | idem |

Le prétraitement est appris **sur le jeu d'entraînement seulement**. Dans la modélisation, il est placé
dans le Pipeline de chaque modèle, et réappris sur chaque pli de la validation croisée.
"""
)

tab_lr, tab_trees = st.tabs(["Régression logistique (standardisée)", "Arbres (unités d'origine)"])
for tab, scale in [(tab_lr, True), (tab_trees, False)]:
    with tab:
        preprocessor = build_preprocessor(numeric, categorical, scale).fit(X_train)
        train_out = preprocessor.transform(X_train)
        test_out = preprocessor.transform(X_test)
        st.markdown(f"**{train_out.shape[1]} colonnes après prétraitement** (aperçu du jeu d'entraînement)")
        st.dataframe(train_out.head(), hide_index=True)
        if scale and numeric:
            check = pd.DataFrame(
                {
                    "moyenne (entraînement)": train_out[numeric].mean(),
                    "écart-type (entraînement)": train_out[numeric].std(ddof=0),
                    "moyenne (test)": test_out[numeric].mean(),
                    "écart-type (test)": test_out[numeric].std(ddof=0),
                }
            ).round(3)
            st.markdown("**Vérification de la standardisation**")
            st.dataframe(check)
            st.caption(
                "Sur l'entraînement : moyenne 0 et écart-type 1 exactement. Sur le test : valeurs proches "
                "mais pas exactes, car le test est transformé avec les statistiques de l'entraînement "
                "(pas de fuite d'information)."
            )
