"""Plotly charts styled to the template: charcoal ink, warm yellow, quiet grids."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

INK = "#303030"
YELLOW = "#FFDD81"
YELLOW_DEEP = "#E9B949"
GRID = "#ECE9E2"
MUTED = "#6B6A66"
FONT = "Work Sans, Helvetica Neue, Arial, sans-serif"
CONFIG = {"displayModeBar": False, "responsive": True}


def _style(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, font=dict(size=12)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=INK, font=dict(family=FONT, color=INK)),
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=12))
    return fig


def fraud_rate_by_type(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("type").agg(rows=("isFraud", "size"), fraud=("isFraud", "sum")).reset_index()
    g["rate"] = g["fraud"] / g["rows"] * 100
    g = g.sort_values("rate", ascending=True)
    colors = [INK if f > 0 else YELLOW for f in g["fraud"]]
    labels = [f"{int(f)} of {int(n):,}" for f, n in zip(g["fraud"], g["rows"])]
    fig = go.Figure(
        go.Bar(
            x=g["rate"], y=g["type"], orientation="h", marker_color=colors,
            text=labels, textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.3f}% fraud<br>%{text} transactions<extra></extra>",
        )
    )
    fig.update_xaxes(title="Fraud rate (%)", rangemode="tozero")
    fig.update_xaxes(range=[0, max(g["rate"].max() * 1.7, 0.1)])
    return _style(fig, 300)


def fraud_by_hour(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("hour").agg(rows=("isFraud", "size"), fraud=("isFraud", "sum")).reindex(range(24), fill_value=0)
    fig = go.Figure()
    fig.add_bar(
        x=g.index, y=g["rows"], name="All transactions", marker_color=YELLOW, yaxis="y",
        hovertemplate="%{x}:00 · %{y:,} transactions<extra></extra>",
    )
    fig.add_scatter(
        x=g.index, y=g["fraud"], name="Fraud cases", mode="lines+markers", yaxis="y2",
        line=dict(color=INK, width=2), marker=dict(size=6, color=INK),
        hovertemplate="%{x}:00 · %{y} fraud cases<extra></extra>",
    )
    fig.update_layout(
        xaxis=dict(title="Hour of day (simulation clock)", tickmode="array", tickvals=list(range(0, 24, 3))),
        yaxis=dict(title="Transactions"),
        yaxis2=dict(title="Fraud cases", overlaying="y", side="right", showgrid=False, rangemode="tozero",
                    tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=12)),
    )
    return _style(fig, 320, legend=True)


def amount_distribution(df: pd.DataFrame) -> go.Figure:
    bins = dict(start=0, end=float(df["amount"].max()) + 1, size=float(df["amount"].max()) / 24)
    fig = go.Figure()
    for label, value, color in (("Legitimate", 0, YELLOW_DEEP), ("Fraud", 1, INK)):
        part = df[df["isFraud"] == value]
        fig.add_histogram(
            x=part["amount"], name=label, histnorm="percent", xbins=bins, marker_color=color,
            opacity=0.85 if value else 0.65,
            hovertemplate=f"{label}: %{{y:.1f}}% of {label.lower()} transactions<extra></extra>",
        )
    fig.update_layout(barmode="overlay", bargap=0.05)
    fig.update_xaxes(title="Amount (after preprocessing cap)")
    fig.update_yaxes(title="Share of group (%)")
    return _style(fig, 320, legend=True)


def balance_behaviour(df: pd.DataFrame) -> go.Figure:
    def shares(part: pd.DataFrame):
        return [
            np.isclose(part["balance_diff_org"], -part["amount"]).mean() * 100,
            (part["balance_diff_org"] < 0).mean() * 100,
            (part["balance_diff_dest"] == 0).mean() * 100,
        ]

    cats = ["Origin fell by the full amount", "Origin balance fell", "Destination balance unchanged"]
    fig = go.Figure()
    for label, value, color in (("Legitimate", 0, YELLOW_DEEP), ("Fraud", 1, INK)):
        fig.add_bar(
            x=shares(df[df["isFraud"] == value]), y=cats, orientation="h", name=label, marker_color=color,
            hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_xaxes(title="Share of transactions (%)", range=[0, 100])
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 300, legend=True)


def threshold_curve(table: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=table["threshold"], y=table["recall"] * 100, name="Fraud caught (recall)", mode="lines+markers",
        line=dict(color=INK, width=2), marker=dict(size=5),
        hovertemplate="threshold %{x:.2f}: %{y:.1f}% of fraud caught<extra></extra>",
    )
    fig.add_scatter(
        x=table["threshold"], y=table["precision"] * 100, name="Alerts that are fraud (precision)",
        mode="lines+markers", line=dict(color=YELLOW_DEEP, width=2, dash="dot"), marker=dict(size=5),
        hovertemplate="threshold %{x:.2f}: %{y:.1f}% of alerts are fraud<extra></extra>",
    )
    fig.add_vline(x=threshold, line_width=1, line_color=INK, line_dash="dash")
    fig.update_xaxes(title="Decision threshold", range=[0, 1])
    fig.update_yaxes(title="%", range=[0, 102])
    return _style(fig, 340, legend=True)
