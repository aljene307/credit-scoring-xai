"""Explicabilité : SHAP, LIME et phrase de synthèse (module 5, choix X01 à X08)."""

import numpy as np
import pandas as pd

from credit_scoring.config import LABELS, PAY_STATUS_LABELS, RANDOM_STATE

MONTHS = {
    "PAY_0": "septembre", "PAY_2": "août", "PAY_3": "juillet", "PAY_4": "juin", "PAY_5": "mai", "PAY_6": "avril",
    "BILL_AMT1": "septembre", "BILL_AMT2": "août", "BILL_AMT3": "juillet", "BILL_AMT4": "juin",
    "BILL_AMT5": "mai", "BILL_AMT6": "avril",
    "PAY_AMT1": "septembre", "PAY_AMT2": "août", "PAY_AMT3": "juillet", "PAY_AMT4": "juin",
    "PAY_AMT5": "mai", "PAY_AMT6": "avril",
}


# --- SHAP -------------------------------------------------------------------------------------

def aggregate_onehot(values: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    """Additionne les colonnes one-hot (EDUCATION_1…4) par variable d'origine (choix X01).

    Valide parce que les valeurs SHAP sont additives : la somme par ligne est inchangée.
    """
    out = pd.DataFrame(index=values.index)
    for feature in features:
        if feature in categorical:
            columns = [c for c in values.columns if c.startswith(f"{feature}_")]
            out[feature] = values[columns].sum(axis=1)
        else:
            out[feature] = values[feature]
    return out


def background_sample(
    X_train: pd.DataFrame, size: int = 200, seed: int = RANDOM_STATE, scores: np.ndarray | None = None
) -> pd.DataFrame:
    """Clients de référence pour SHAP en probabilité, tirés dans l'entraînement.

    Sans `scores` : tirage simple. Avec `scores` (probabilités prédites sur l'entraînement) :
    tirage stratifié, le même nombre de clients dans chacun des 10 déciles de score, pour que
    la valeur de base (probabilité moyenne des clients de référence) reste proche de la
    probabilité moyenne réelle quel que soit le tirage (choix X03).
    """
    rng = np.random.default_rng(seed)
    size = min(size, len(X_train))
    if scores is None:
        return X_train.iloc[rng.choice(len(X_train), size=size, replace=False)]
    deciles = pd.qcut(pd.Series(scores, index=X_train.index), 10, labels=False, duplicates="drop")
    per_group = size // deciles.nunique()
    chosen = [rng.choice(np.flatnonzero(deciles.to_numpy() == d), per_group, replace=False) for d in sorted(deciles.unique())]
    return X_train.iloc[np.sort(np.concatenate(chosen))]


class ShapExplainer:
    """Valeurs SHAP par variable d'origine, calculées exactement (choix X02).

    - Modèles à base d'arbres (XGBoost, arbre, Random Forest) : TreeExplainer
      « interventionnel » sur la sortie en **probabilité** (`scale = "probabilité"`).
    - Régression logistique : LinearExplainer, exact, sur la sortie en **log-odds**
      (`scale = "log-odds"`) ; aucune méthode exacte n'existe en probabilité pour ce
      modèle, et la permutation mesurée est bruitée (jusqu'à 1,1 point d'écart entre deux
      calculs du même client).
    - Autres modèles : PermutationExplainer (approché), en probabilité.
    Dans tous les cas : sortie du modèle = valeur de base + somme des valeurs SHAP.
    """

    def __init__(self, pipeline, background: pd.DataFrame, categorical: list[str], max_evals: int = 500):
        import shap

        self.pipeline = pipeline
        self.max_evals = max_evals  # PermutationExplainer seulement : nombre d'évaluations par client
        self.features = list(background.columns)
        self.categorical = categorical
        estimator = pipeline[-1]
        self._pre = pipeline[:-1]
        self.scale = "probabilité"
        if hasattr(estimator, "get_booster") or hasattr(estimator, "estimators_") or hasattr(estimator, "tree_"):
            self.method = "tree"
            # Un masker avec max_samples explicite : passé directement, l'échantillon serait réduit
            # en silence à 100 clients tirés au hasard (mesuré avec shap 0.51.0, choix X03).
            masker = shap.maskers.Independent(self._pre.transform(background), max_samples=len(background))
            self._explainer = shap.TreeExplainer(
                estimator, data=masker, model_output="probability", feature_perturbation="interventional"
            )
        elif hasattr(estimator, "coef_"):
            self.method = "linear"
            self.scale = "log-odds"
            self._explainer = shap.LinearExplainer(estimator, self._pre.transform(background))
        else:
            self.method = "permutation"
            masker = shap.maskers.Independent(background, max_samples=len(background))
            self._explainer = shap.PermutationExplainer(self._predict, masker, seed=RANDOM_STATE)

    def _predict(self, X) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(X), columns=self.features)
        return self.pipeline.predict_proba(_restore_types(frame, self.categorical))[:, 1]

    def explain(self, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Valeurs SHAP (une ligne par client, une colonne par variable d'origine) et valeur de base."""
        if self.method in ("tree", "linear"):
            transformed = self._pre.transform(X)
            result = self._explainer(transformed)
            raw, bases = result.values, np.asarray(result.base_values)
            if raw.ndim == 3:  # modèles scikit-learn : une valeur par classe, on garde « défaut »
                raw, bases = raw[..., 1], bases[..., 1]
            values = pd.DataFrame(raw, index=X.index, columns=transformed.columns)
            values = aggregate_onehot(values, self.features, self.categorical)
        else:
            result = self._explainer(X, max_evals=self.max_evals, silent=True)
            values = pd.DataFrame(result.values, index=X.index, columns=self.features)
            bases = np.asarray(result.base_values)
        return values, float(np.ravel(bases)[0])


IMPORTANCE_LABELS = {
    "logreg": "|coefficient| par écart-type (numériques) ; écart entre modalités extrêmes (catégorielles)",
    "arbre": "part de la réduction d'impureté",
    "random_forest": "part de la réduction d'impureté",
    "xgboost": "part du gain total",
}


def model_importance(pipeline, features: list[str], categorical: list[str]) -> pd.Series:
    """Importance propre au modèle, par variable d'origine (choix X12).

    - Arbre, Random Forest : part de la réduction d'impureté ; XGBoost : part du gain
      total. Ces quantités s'additionnent : la somme des colonnes one-hot est valide.
    - Régression logistique : |coefficient| pour une variable numérique (effet d'un
      écart-type) ; pour une catégorielle, écart entre le coefficient de la modalité la
      plus risquée et celui de la moins risquée (seules les différences entre modalités
      ont un sens). Les deux unités diffèrent, et l'écart entre modalités ne tient pas
      compte du nombre de clients concernés.
    Non comparable d'un modèle à l'autre : d'où l'importance par permutation.
    """
    estimator = pipeline[-1]
    columns = list(pipeline[:-1].get_feature_names_out())
    if hasattr(estimator, "coef_"):
        coef = pd.Series(estimator.coef_[0], index=columns)
        importance = {}
        for feature in features:
            if feature in categorical:
                levels = coef[[c for c in columns if c.startswith(f"{feature}_")]]
                importance[feature] = float(levels.max() - levels.min())
            else:
                importance[feature] = float(abs(coef[feature]))
        return pd.Series(importance).sort_values(ascending=False)
    if hasattr(estimator, "get_booster"):
        total_gain = estimator.get_booster().get_score(importance_type="total_gain")
        raw = np.array([total_gain.get(c, 0.0) for c in columns])
        raw = raw / raw.sum()
    else:
        raw = estimator.feature_importances_
    per_column = pd.DataFrame([raw], columns=columns)
    return aggregate_onehot(per_column, features, categorical).iloc[0].sort_values(ascending=False)


# --- LIME -----------------------------------------------------------------------------------------

def _restore_types(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """LIME et SHAP passent des tableaux de flottants : les codes catégoriels redeviennent entiers."""
    frame = frame.copy()
    for col in categorical:
        frame[col] = frame[col].round().astype(int)
    return frame


class LimeExplainer:
    """Explication LIME sur les 23 variables d'origine (choix X05)."""

    def __init__(self, X_train: pd.DataFrame, categorical: list[str]):
        self.features = list(X_train.columns)
        self.categorical = categorical
        self._training = X_train.to_numpy(dtype=float)

    def explain(self, pipeline, client: pd.Series, num_samples: int, seed: int, num_features: int = 10):
        """Poids LIME par variable (signe = sens de l'effet sur le risque) et règles lisibles."""
        from lime.lime_tabular import LimeTabularExplainer

        explainer = LimeTabularExplainer(
            self._training,
            feature_names=self.features,
            categorical_features=[self.features.index(c) for c in self.categorical],
            class_names=["Pas de défaut", "Défaut"],
            mode="classification",
            discretize_continuous=True,
            discretizer="quartile",
            random_state=seed,
        )

        def predict(array: np.ndarray) -> np.ndarray:
            frame = _restore_types(pd.DataFrame(array, columns=self.features), self.categorical)
            return pipeline.predict_proba(frame)

        explanation = explainer.explain_instance(
            client.to_numpy(dtype=float), predict, labels=(1,), num_features=num_features, num_samples=num_samples
        )
        weights = pd.Series({self.features[i]: w for i, w in explanation.as_map()[1]})
        rules = pd.Series(dict(explanation.as_list(label=1)))
        return weights.reindex(weights.abs().sort_values(ascending=False).index), rules


def top_features(contributions: pd.Series, k: int = 5) -> list[str]:
    """Les k variables de plus forte contribution, en valeur absolue."""
    return list(contributions.abs().sort_values(ascending=False).index[:k])


def agreement(shap_row: pd.Series, lime_weights: pd.Series, k: int = 5) -> dict:
    """Variables communes aux k premières de SHAP et de LIME, et accord de leurs signes (choix X06)."""
    common = set(top_features(shap_row, k)) & set(top_features(lime_weights, k))
    same_sign = sum(np.sign(shap_row[f]) == np.sign(lime_weights[f]) for f in common)
    return {"communes": len(common), "meme_signe": int(same_sign), "variables": sorted(common)}


# --- Phrase de synthèse -------------------------------------------------------------------------

def _amount(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ") + " NT$"


def describe(feature: str, value, median_limit: float | None = None) -> str:
    """Valeur d'une variable en langage courant (ex. PAY_0 = 2 -> « un retard de 2 mois en septembre »)."""
    if feature.startswith("PAY_") and not feature.startswith("PAY_AMT"):
        code = int(value)
        if code >= 1:
            return f"un retard de paiement de {code} mois en {MONTHS[feature]}"
        return {
            -2: f"aucune utilisation de la carte en {MONTHS[feature]}",
            -1: f"un paiement en totalité en {MONTHS[feature]}",
            0: f"un paiement du minimum en {MONTHS[feature]}",
        }.get(code, f"le statut {PAY_STATUS_LABELS.get(code, code)} en {MONTHS[feature]}")
    if feature == "LIMIT_BAL":
        level = "" if median_limit is None else (" faible" if value < median_limit else " élevé")
        return f"un plafond de crédit{level} ({_amount(value)})"
    if feature.startswith("BILL_AMT"):
        return f"une facture de {_amount(value)} en {MONTHS[feature]}"
    if feature.startswith("PAY_AMT"):
        return f"un paiement de {_amount(value)} en {MONTHS[feature]}"
    if feature == "AGE":
        return f"son âge ({int(value)} ans)"
    if feature in LABELS:
        noun = {"SEX": "son sexe", "EDUCATION": "son niveau d'études", "MARRIAGE": "sa situation familiale"}[feature]
        return f"{noun} ({LABELS[feature].get(int(value), value).lower()})"
    return f"{feature} = {value}"


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " et " + parts[-1]


def _de(text: str) -> str:
    """« de » avec élision : d'un retard, d'une facture, d'aucune utilisation, de son âge."""
    return ("d'" + text) if text.startswith(("un ", "une ", "aucune ")) else ("de " + text)


def summary_sentence(
    contributions: pd.Series,
    client: pd.Series,
    proba: float,
    base: float,
    threshold: float,
    median_limit: float | None = None,
    k: int = 3,
) -> str:
    """Phrase pour un décideur, à partir des k plus fortes contributions SHAP dans le sens de la décision.

    Les statuts de remboursement identiques sur plusieurs mois sont regroupés
    (« un retard de 2 mois en septembre et en août »).
    """
    above = proba >= base
    direction = contributions if above else -contributions
    drivers = [f for f in direction.sort_values(ascending=False).index[:k] if direction[f] > 0]
    # élevé : client signalé au seuil ; modéré : au-dessus de la moyenne sans être signalé.
    level = "élevé" if proba >= threshold else ("modéré" if above else "faible")
    head = f"Le risque de défaut est {level} ({proba:.0%} contre {base:.0%} en moyenne)".replace("%", " %")
    if not drivers:
        return head + "."

    parts, done = [], set()
    for feature in drivers:
        if feature in done:
            continue
        if feature.startswith("PAY_") and not feature.startswith("PAY_AMT"):
            same = [f for f in drivers if f.startswith("PAY_") and not f.startswith("PAY_AMT")
                    and int(client[f]) == int(client[feature]) and f not in done]
            same.sort(key=list(MONTHS).index)  # du mois le plus récent au plus ancien
            text = describe(feature, client[feature])
            if len(same) > 1:
                text = text.rsplit(" en ", 1)[0] + " " + _join([f"en {MONTHS[f]}" for f in same])
            done.update(same)
            parts.append(text)
        else:
            done.add(feature)
            parts.append(describe(feature, client[feature], median_limit))
    if above:
        cause = "à cause " + _join([_de(p) for p in parts])
    else:
        cause = "grâce " + _join(["à " + p for p in parts])
    return f"{head}, principalement {cause}."
