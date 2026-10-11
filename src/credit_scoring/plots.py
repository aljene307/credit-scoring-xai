"""Graphiques Plotly de l'application (couleurs : choix C11)."""

import pandas as pd
import plotly.graph_objects as go

from credit_scoring.eda import display_value, is_discrete

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
