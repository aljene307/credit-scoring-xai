"""Graphiques Plotly de l'application (couleurs : choix C11)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

from credit_scoring.eda import display_value, is_discrete
from credit_scoring.evaluation import confusion_counts, metrics_at_threshold

CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}  # bleu : classe 0, orange : classe 1
RATE_COLOR = "#2a78d6"
MUTED = "#898781"
# Échelle divergente bleu <-> gris <-> rouge, centrée sur 0.
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec8483"], [1.0, "#b8302f"]]


def _layout(fig: go.Figure, title: str, x_title: str = "", y_title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        barcornerradius=4,
        bargap=0.25,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 70, "l": 10, "r": 10, "b": 10},
        hoverlabel={"namelength": -1},
    )
    return fig


def class_balance_figure(balance: pd.DataFrame, class_names: dict) -> go.Figure:
    """Barres : effectif de chaque classe, part en étiquette."""
    fig = go.Figure(
        go.Bar(
            x=[class_names[c] for c in balance.index],
            y=balance["clients"],
            marker_color=[CLASS_COLORS[c] for c in balance.index],
            text=[f"{p:.1f} %" for p in balance["part (%)"]],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,} lignes<extra></extra>",
        )
    )
    return _layout(fig, "Répartition des classes", y_title="Nombre de lignes")


def distribution_figure(
    df: pd.DataFrame, col: str, target: str, categorical: bool, class_names: dict, normalize: bool
) -> go.Figure:
    """Distribution d'une variable, séparée par classe.

    normalize=True : pourcentage à l'intérieur de chaque classe, pour comparer
    les formes malgré le déséquilibre des classes.
    """
    fig = go.Figure()
    y_title = "% de la classe" if normalize else "Nombre de lignes"
    if is_discrete(df[col], categorical):
        for cls in sorted(df[target].unique()):
            counts = df.loc[df[target] == cls, col].value_counts(normalize=normalize).sort_index()
            values = counts * 100 if normalize else counts
            fig.add_bar(
                x=[display_value(col, v) for v in counts.index],
                y=values,
                name=class_names[cls],
                marker_color=CLASS_COLORS[cls],
                hovertemplate="%{x}<br>%{y:,.1f}<extra>%{fullData.name}</extra>"
                if normalize
                else "%{x}<br>%{y:,}<extra>%{fullData.name}</extra>",
            )
        fig.update_layout(barmode="group")
        fig.update_xaxes(type="category")
    else:
        for cls in sorted(df[target].unique()):
            fig.add_histogram(
                x=df.loc[df[target] == cls, col],
                name=class_names[cls],
                marker_color=CLASS_COLORS[cls],
                opacity=0.6,
                nbinsx=50,
                histnorm="percent" if normalize else "",
            )
        fig.update_layout(barmode="overlay")
    return _layout(fig, f"Distribution de {col} par classe", x_title=col, y_title=y_title)


def positive_rate_figure(table: pd.DataFrame, col: str, overall_rate: float, positive_name: str) -> go.Figure:
    """Barres : taux de la classe positive par modalité ou tranche, avec la moyenne."""
    fig = go.Figure(
        go.Bar(
            x=table.index.astype(str),
            y=table["taux (%)"],
            marker_color=RATE_COLOR,
            customdata=table["clients"],
            hovertemplate="%{x}<br>Taux : %{y:.1f} %<br>%{customdata:,} lignes<extra></extra>",
        )
    )
    fig.add_hline(
        y=overall_rate,
        line_dash="dash",
        line_color=MUTED,
        annotation_text=f"Moyenne : {overall_rate:.1f} %",
        annotation_position="top left",
    )
    fig.update_xaxes(type="category")
    return _layout(fig, f"Taux de « {positive_name} » selon {col}", x_title=col, y_title="Taux (%)")


# Couleur fixe par modèle (ordre de la palette catégorielle) : la référence prend la première.
MODEL_COLORS = {"xgboost": "#2a78d6", "random_forest": "#eb6834", "arbre": "#1baf7a", "logreg": "#eda100"}
SEQUENTIAL = [[0.0, "#f0efec"], [1.0, "#1c5cab"]]


def roc_figure(y_true, probas: dict, labels: dict, thresholds: dict) -> go.Figure:
    """Courbes ROC superposées ; un point marque le seuil retenu de chaque modèle."""
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": MUTED, "dash": "dash", "width": 1},
                    name="hasard (AUC 0,5)", hoverinfo="skip")
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        color = MODEL_COLORS.get(name, RATE_COLOR)
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line={"color": color, "width": 2}, name=labels[name],
                        hovertemplate="faux positifs %{x:.1%}<br>défauts détectés %{y:.1%}<extra>%{fullData.name}</extra>")
        counts = confusion_counts(y_true, proba, thresholds[name])
        point_fpr = counts["fp"] / (counts["fp"] + counts["vn"])
        point_tpr = counts["vp"] / (counts["vp"] + counts["fn"])
        fig.add_scatter(x=[point_fpr], y=[point_tpr], mode="markers", showlegend=False,
                        marker={"size": 10, "color": color, "line": {"color": "white", "width": 2}},
                        hovertemplate=f"{labels[name]} au seuil {thresholds[name]:.3f}<extra></extra>")
    fig = _layout(fig, "Courbes ROC (test)", x_title="Taux de bons clients signalés à tort",
                  y_title="Taux de défauts détectés (rappel)")
    fig.update_layout(height=480)
    return fig


def precision_recall_figure(y_true, probas: dict, labels: dict, thresholds: dict) -> go.Figure:
    """Courbes précision-rappel superposées, avec la ligne du hasard (taux de défaut)."""
    fig = go.Figure()
    base = float(np.mean(y_true))
    fig.add_scatter(x=[0, 1], y=[base, base], mode="lines", line={"color": MUTED, "dash": "dash", "width": 1},
                    name=f"hasard ({base:.0%})", hoverinfo="skip")
    for name, proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        color = MODEL_COLORS.get(name, RATE_COLOR)
        fig.add_scatter(x=recall, y=precision, mode="lines", line={"color": color, "width": 2}, name=labels[name],
                        hovertemplate="rappel %{x:.1%}<br>précision %{y:.1%}<extra>%{fullData.name}</extra>")
        m = metrics_at_threshold(y_true, proba, thresholds[name])
        fig.add_scatter(x=[m["rappel"]], y=[m["precision"]], mode="markers", showlegend=False,
                        marker={"size": 10, "color": color, "line": {"color": "white", "width": 2}},
                        hovertemplate=f"{labels[name]} au seuil {thresholds[name]:.3f}<extra></extra>")
    fig = _layout(fig, "Courbes précision-rappel (test)", x_title="Rappel (défauts détectés)",
                  y_title="Précision (vrais défauts parmi les signalés)")
    fig.update_layout(height=480, yaxis_range=[0, 1.02])
    return fig


def confusion_figure(counts: dict) -> go.Figure:
    """Matrice de confusion : effectifs et part du total."""
    total = sum(counts.values())
    z = [[counts["vp"], counts["fn"]], [counts["fp"], counts["vn"]]]
    names = [["vrais positifs", "défauts manqués"], ["bons clients signalés", "vrais négatifs"]]
    text = [[f"{names[i][j]}<br><b>{z[i][j]:,}</b><br>{z[i][j] / total:.1%}".replace(",", " ") for j in range(2)]
            for i in range(2)]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=["Prédit « défaut »", "Prédit « pas de défaut »"], y=["Défaut réel", "Pas de défaut réel"],
            text=text, texttemplate="%{text}", colorscale=SEQUENTIAL, showscale=False,
            hovertemplate="%{y} / %{x}<br>%{z:,} clients<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig = _layout(fig, "Matrice de confusion (test)")
    fig.update_layout(height=340)
    return fig


def ks_figure(y_true, proba, class_names: dict) -> go.Figure:
    """Courbes cumulées des scores par classe ; le KS est leur plus grand écart vertical."""
    y = np.asarray(y_true)
    p = np.asarray(proba)
    # Scores réellement observés : l'écart maximal est exactement le KS du tableau.
    grid = np.concatenate([[0.0], np.unique(p)])
    cum = {cls: np.searchsorted(np.sort(p[y == cls]), grid, side="right") / (y == cls).sum() for cls in (0, 1)}
    gap = cum[0] - cum[1]
    best = int(np.argmax(gap))
    fig = go.Figure()
    for cls in (0, 1):
        fig.add_scatter(x=grid, y=cum[cls], mode="lines", name=class_names[cls],
                        line={"color": CLASS_COLORS[cls], "width": 2},
                        hovertemplate="score ≤ %{x:.2f}<br>%{y:.1%} des clients<extra>%{fullData.name}</extra>")
    fig.add_scatter(x=[grid[best], grid[best]], y=[cum[1][best], cum[0][best]], mode="lines+markers",
                    line={"color": MUTED, "dash": "dot", "width": 2}, marker={"size": 8, "color": MUTED},
                    name=f"KS = {gap[best]:.3f} (score {grid[best]:.2f})")
    fig = _layout(fig, "Statistique de Kolmogorov-Smirnov (test)", x_title="Probabilité de défaut prédite",
                  y_title="Part cumulée des clients")
    fig.update_layout(height=420, yaxis_tickformat=".0%")
    return fig


def calibration_figure(table: pd.DataFrame, label: str) -> go.Figure:
    """Taux de défaut observé contre probabilité moyenne prédite, par tranche de même effectif."""
    top = float(max(table["proba_moyenne"].max(), table["taux_observe"].max())) * 1.05
    fig = go.Figure()
    fig.add_scatter(x=[0, top], y=[0, top], mode="lines", line={"color": MUTED, "dash": "dash", "width": 1},
                    name="calibration parfaite", hoverinfo="skip")
    fig.add_scatter(x=table["proba_moyenne"], y=table["taux_observe"], mode="lines+markers", name=label,
                    line={"color": RATE_COLOR, "width": 2}, marker={"size": 9, "color": RATE_COLOR},
                    customdata=table["clients"],
                    hovertemplate="prédit %{x:.1%}<br>observé %{y:.1%}<br>%{customdata} clients<extra></extra>")
    fig = _layout(fig, f"Calibration : {label} (test, 10 tranches de même effectif)",
                  x_title="Probabilité moyenne prédite", y_title="Taux de défaut observé")
    fig.update_layout(height=440, xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    return fig


def auc_comparison_figure(table: pd.DataFrame, reference_label: str | None = None) -> go.Figure:
    """Points : AUC de validation croisée de chaque modèle, barres d'erreur = écart-type entre plis.

    Des points plutôt que des barres : une barre devrait partir de 0, ce qui
    écraserait des écarts de quelques centièmes.
    """
    ordered = table.sort_values("AUC")
    labels = [
        f"{m} (référence)" if m == reference_label else m for m in ordered["modèle"]
    ]
    fig = go.Figure(
        go.Scatter(
            x=ordered["AUC"],
            y=labels,
            mode="markers",
            marker={"size": 12, "color": RATE_COLOR},
            error_x={"type": "data", "array": ordered["écart-type AUC"], "color": MUTED, "thickness": 2},
            hovertemplate="%{y}<br>AUC : %{x:.4f} ± %{error_x.array:.4f}<extra></extra>",
        )
    )
    fig = _layout(fig, "AUC de validation croisée (± écart-type entre les 5 plis)", x_title="AUC")
    fig.update_layout(height=320)
    return fig


def correlation_figure(corr: pd.DataFrame, method_name: str) -> go.Figure:
    """Carte des corrélations, échelle divergente centrée sur 0."""
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale=DIVERGING,
            hovertemplate="%{y} / %{x}<br>corrélation : %{z:.2f}<extra></extra>",
            colorbar={"title": "corr."},
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig = _layout(fig, f"Corrélations ({method_name})")
    fig.update_layout(height=650)
    return fig
