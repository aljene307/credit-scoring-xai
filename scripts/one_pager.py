"""Produit le one-pager (HTML puis PDF A4) à partir du gabarit et des résultats du projet.

Usage, depuis la racine du projet (après scripts/train.py et scripts/expliquer.py) :
    .venv\\Scripts\\python.exe scripts\\one_pager.py

Les chiffres sont lus dans models/ ou recalculés sur le jeu de test, jamais recopiés à la main ;
le texte se modifie dans docs/one-pager/one-pager.template.html. Le PDF est imprimé par
Microsoft Edge en mode sans fenêtre (aucun outil supplémentaire).
"""

import base64
import io
import json
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from string import Template

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_scoring.comparison import holdout_table
from credit_scoring.config import BILL_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR, PAY_AMT_FEATURES, PROJECT_ROOT
from credit_scoring.evaluation import bootstrap_auc, confusion_counts
from credit_scoring.explain import (LIME_SAMPLES, LimeExplainer, ShapExplainer, agreement, summary_sentence)
from credit_scoring.global_explanations import API_BUNDLE
from credit_scoring.reference import load_reference, reference_split
from credit_scoring.training import load_model

FOLDER = PROJECT_ROOT / "docs" / "one-pager"
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]


def fr(value: float, decimals: int = 3) -> str:
    """Nombre au format français : virgule décimale, espace pour les milliers."""
    text = f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return text


def figure_pay0(shap_values: pd.DataFrame, X: pd.DataFrame) -> str:
    """Petite figure : contribution SHAP moyenne de PAY_0 par statut (points de probabilité)."""
    values = X.loc[shap_values.index, "PAY_0"]
    means = (shap_values["PAY_0"] * 100).groupby(values).agg(["mean", "size"])
    means = means[means["size"] >= 30]  # statuts assez fréquents pour une moyenne stable
    labels = {-2: "pas\nd'utilisation", -1: "payé en\ntotalité", 0: "minimum\npayé", 1: "1 mois\nde retard",
              2: "2 mois\nde retard"}
    fig, ax = plt.subplots(figsize=(3.6, 1.55), dpi=200)
    colors = ["#eb6834" if v > 0 else "#2a78d6" for v in means["mean"]]
    ax.bar([labels.get(int(v), str(v)) for v in means.index], means["mean"], color=colors, width=0.62)
    for i, v in enumerate(means["mean"]):
        ax.annotate(f"{v:+.1f}".replace(".", ","), (i, v), ha="center", va="bottom" if v > 0 else "top",
                    fontsize=6, color="#52514e", xytext=(0, 1.5 if v > 0 else -1.5), textcoords="offset points")
    ax.axhline(0, color="#898781", linewidth=0.6)
    ax.set_ylabel("points", fontsize=6.5, color="#52514e")
    ax.tick_params(labelsize=5.8, colors="#52514e", length=0)
    ax.set_ylim(min(means["mean"].min() * 1.6, -2), means["mean"].max() * 1.22)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    fig.tight_layout(pad=0.3)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    warnings.filterwarnings("ignore")
    results, models = load_reference()
    X_train, X_test, y_train, y_test = reference_split(results["meta"])
    summaries = results["modeles"]
    bundle = load_model(MODELS_DIR / API_BUNDLE)
    explications = json.loads((MODELS_DIR / "explications.json").read_text(encoding="utf-8"))

    # Résultats sur le test
    probas = {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}
    table = holdout_table(y_test, probas, summaries, bootstrap_auc(y_test, probas, n_resamples=1000)).set_index("_cle")
    threshold = summaries["xgboost"]["seuils"]["f1_max"]["seuil"]
    counts = confusion_counts(y_test, probas["xgboost"], threshold)
    sans = bundle["variantes"]["sans_sensibles"]

    # Explications
    shap_values = pd.read_parquet(MODELS_DIR / "shap_xgboost.parquet")
    pay0 = (shap_values["PAY_0"] * 100).groupby(X_test.loc[shap_values.index, "PAY_0"]).mean()
    complet = bundle["variantes"]["complet"]
    shap_xgb = ShapExplainer(models["xgboost"], complet["fond"], CATEGORICAL_FEATURES)
    base_proba = float(models["xgboost"].predict_proba(complet["fond"])[:, 1].mean())
    client = X_test.loc[[1358]]
    contributions = shap_xgb.explain(client)[0].iloc[0]
    phrase = summary_sentence(contributions, client.iloc[0], probas["xgboost"][X_test.index.get_loc(1358)],
                              base_proba, threshold, bundle["mediane_plafond"])

    # Accord SHAP / LIME sur 20 clients du test (même tirage que la mesure X08)
    clients = X_test.iloc[np.random.default_rng(42).choice(len(X_test), 20, replace=False)]
    lime = LimeExplainer.from_stats(complet["lime"], complet["variables"], CATEGORICAL_FEATURES)
    values20 = shap_xgb.explain(clients)[0]
    accords = [agreement(values20.loc[i], lime.explain(models["xgboost"], row, num_samples=LIME_SAMPLES, seed=0)[0])
               for i, row in clients.iterrows()]
    communes = sum(a["communes"] for a in accords)

    # Équité, compromis, limites
    groupes = explications["equite"]["groupes"]
    sexe = {r["groupe"]: r for r in groupes["sexe"]["avec"]["tableau"]}
    age = {r["groupe"]: r for r in groupes["âge"]["avec"]["tableau"]}
    perte = (explications["equite"]["auc_validation_sans"]["moyenne"]
             - explications["equite"]["auc_validation_avec"]["moyenne"])
    grille_arbre = pd.read_parquet(MODELS_DIR / "grille_arbre.parquet")
    arbre3 = grille_arbre.loc[grille_arbre["max_depth"] == 3, "mean_test_roc_auc"].max()
    xgb_cv = summaries["xgboost"]["validation_croisee"]["roc_auc"]["moyenne"]
    data = pd.concat([X_train, X_test])
    target = pd.concat([y_train, y_test])
    inactifs = data[BILL_FEATURES + PAY_AMT_FEATURES].eq(0).all(axis=1)

    values = {
        "n_clients": fr(len(data), 0), "taux_defaut": fr(target.mean() * 100, 1), "n_test": fr(len(X_test), 0),
        "combinaisons": sum(s["combinaisons"] for s in summaries.values()),
        "auc_xgboost": fr(table.loc["xgboost", "AUC test"]), "ic_bas": fr(table.loc["xgboost", "IC 95 % bas"]),
        "ic_haut": fr(table.loc["xgboost", "IC 95 % haut"]),
        "gini": fr(table.loc["xgboost", "Gini"]), "ks": fr(table.loc["xgboost", "KS"]),
        "auc_random_forest": fr(table.loc["random_forest", "AUC test"]), "auc_arbre": fr(table.loc["arbre", "AUC test"]),
        "auc_logreg": fr(table.loc["logreg", "AUC test"]), "auc_sans": fr(sans["auc_test"]),
        "seuil_sans": fr(sans["seuil"]),
        "signales": fr(counts["vp"] + counts["fp"], 0), "vrais_defauts": fr(counts["vp"], 0),
        "part_vrais": fr(counts["vp"] / (counts["vp"] + counts["fp"]) * 100, 0), "manques": fr(counts["fn"], 0),
        "defauts": fr(counts["vp"] + counts["fn"], 0),
        "shap_pay0_2": fr(pay0.loc[2], 0), "figure_pay0": figure_pay0(shap_values, X_test),
        "phrase_1358": phrase,
        "accord": fr(communes / len(accords), 1),
        "signes": f"{sum(a['meme_signe'] for a in accords)} cas sur {communes}",
        "fp_homme": fr(sexe["Homme"]["faux positifs (%)"], 1), "fp_femme": fr(sexe["Femme"]["faux positifs (%)"], 1),
        "fp_jeunes": fr(age["21-25 ans"]["faux positifs (%)"], 1), "fp_ages": fr(age["56 ans et +"]["faux positifs (%)"], 1),
        "perte_auc": fr(abs(perte)),
        "ecart_logreg": fr(table.loc["xgboost", "AUC test"] - table.loc["logreg", "AUC test"]),
        "ecart_arbre3": fr(xgb_cv - arbre3),
        "inactifs": fr(int(inactifs.sum()), 0), "taux_inactifs": fr(target[inactifs].mean() * 100, 1),
    }
    template = Template((FOLDER / "one-pager.template.html").read_text(encoding="utf-8"))
    html = template.substitute(values)  # erreur si un repère du gabarit n'a pas de valeur
    html_path = FOLDER / "one-pager.html"
    html_path.write_text(html, encoding="utf-8")

    edge = shutil.which("msedge") or next((p for p in EDGE if Path(p).exists()), None)
    if edge is None:
        raise SystemExit("Microsoft Edge introuvable : ouvrez one-pager.html et imprimez-le en PDF.")
    pdf_path = FOLDER / "one-pager.pdf"
    subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()], check=True, timeout=120)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))
    print(f"one-pager : {html_path.name} et {pdf_path.name} ({pdf_path.stat().st_size / 1e3:.0f} Ko, {pages} page(s))")
    for key in ("auc_xgboost", "gini", "ks","auc_sans", "seuil_sans", "accord", "signes", "perte_auc", "ecart_logreg", "ecart_arbre3"):
        print(f"  {key} = {values[key]}")


if __name__ == "__main__":
    main()
