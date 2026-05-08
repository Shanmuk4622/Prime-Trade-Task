"""
visualisation.py
================
Reusable chart builders using Plotly (interactive) and Matplotlib/Seaborn (static).
All chart functions return a figure object that can be shown or saved independently.

Colour palette
--------------
Extreme Fear  → #EF4444  (red)
Fear          → #F97316  (orange)
Neutral       → #EAB308  (yellow)
Greed         → #22C55E  (green)
Extreme Greed → #3B82F6  (blue)
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Regime palette ─────────────────────────────────────────────────────────────
REGIME_COLORS = {
    "Extreme Fear":  "#EF4444",
    "Fear":          "#F97316",
    "Neutral":       "#EAB308",
    "Greed":         "#22C55E",
    "Extreme Greed": "#3B82F6",
}
REGIME_ORDER = list(REGIME_COLORS.keys())

# ── Plotly template ────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sentiment over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_sentiment_timeline(sentiment_df: pd.DataFrame, save: bool = True):
    """Line chart of Fear/Greed value over time, coloured by regime."""
    df = sentiment_df.dropna(subset=["trade_date"]).copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    fig = px.line(
        df, x="trade_date", y="value" if "value" in df.columns else "regime_numeric",
        color="regime",
        color_discrete_map=REGIME_COLORS,
        category_orders={"regime": REGIME_ORDER},
        title="Bitcoin Fear & Greed Index — Timeline",
        labels={"trade_date": "Date", "value": "Fear/Greed Score"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(hovermode="x unified", legend_title="Sentiment Regime")

    if save:
        path = FIG_DIR / "01_sentiment_timeline.html"
        fig.write_html(str(path))
        print(f"[📊] Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. PnL distribution by regime
# ─────────────────────────────────────────────────────────────────────────────

def plot_pnl_by_regime(df: pd.DataFrame, pnl_col: str = "closed_pnl", save: bool = True):
    """Violin + box overlay: PnL distribution for each sentiment regime."""
    plot_df = df.dropna(subset=[pnl_col, "regime"]).copy()
    # Clip extreme outliers for readability
    q01, q99 = plot_df[pnl_col].quantile([0.01, 0.99])
    plot_df = plot_df[(plot_df[pnl_col] >= q01) & (plot_df[pnl_col] <= q99)]

    fig = px.violin(
        plot_df,
        x="regime", y=pnl_col,
        color="regime",
        box=True,
        points=False,
        color_discrete_map=REGIME_COLORS,
        category_orders={"regime": REGIME_ORDER},
        title="Trader PnL Distribution by Sentiment Regime",
        labels={"regime": "Sentiment Regime", pnl_col: "Closed PnL (USD)"},
        template=PLOTLY_TEMPLATE,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig.update_layout(showlegend=False)

    if save:
        path = FIG_DIR / "02_pnl_by_regime.html"
        fig.write_html(str(path))
        print(f"[📊] Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Win rate by regime
# ─────────────────────────────────────────────────────────────────────────────

def plot_winrate_by_regime(regime_stats: pd.DataFrame, save: bool = True):
    """Bar chart of win rates across regimes."""
    df = regime_stats.reset_index()

    fig = px.bar(
        df, x="regime", y="win_rate",
        color="regime",
        color_discrete_map=REGIME_COLORS,
        category_orders={"regime": REGIME_ORDER},
        text=df["win_rate"].map(lambda v: f"{v*100:.1f}%"),
        title="Win Rate by Sentiment Regime",
        labels={"regime": "Sentiment Regime", "win_rate": "Win Rate"},
        template=PLOTLY_TEMPLATE,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.5,
                  annotation_text="50% baseline")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%")

    if save:
        path = FIG_DIR / "03_winrate_by_regime.html"
        fig.write_html(str(path))
        print(f"[📊] Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str] = None, save: bool = True):
    """Seaborn heatmap of Spearman correlations between key numeric columns."""
    if cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        key_cols = [c for c in numeric_cols if any(
            k in c for k in ["pnl", "win_rate", "regime_numeric", "value", "leverage", "size"]
        )]
        cols = key_cols[:12]  # Cap at 12 for readability

    corr = df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Spearman Correlation Matrix — Key Variables", fontsize=14, pad=15)
    plt.tight_layout()

    if save:
        path = FIG_DIR / "04_correlation_heatmap.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        print(f"[📊] Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Top account PnL bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_accounts(account_summary: pd.DataFrame, n: int = 20, save: bool = True):
    """Horizontal bar chart of the top-N and bottom-N accounts by total PnL."""
    top_n = account_summary.nlargest(n, "total_pnl")
    bot_n = account_summary.nsmallest(n, "total_pnl")
    combined = pd.concat([top_n, bot_n])
    combined["colour"] = combined["total_pnl"].apply(lambda v: "#22C55E" if v > 0 else "#EF4444")
    combined["account_short"] = combined.index.astype(str).str[:12] + "…"

    fig = go.Figure(go.Bar(
        x=combined["total_pnl"],
        y=combined["account_short"],
        orientation="h",
        marker_color=combined["colour"],
    ))
    fig.update_layout(
        title=f"Top {n} & Bottom {n} Accounts by Total PnL",
        xaxis_title="Total Closed PnL (USD)",
        yaxis_title="Account",
        template=PLOTLY_TEMPLATE,
        height=800,
    )

    if save:
        path = FIG_DIR / "05_top_accounts_pnl.html"
        fig.write_html(str(path))
        print(f"[📊] Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. PCA cluster scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_clusters(pca_df: pd.DataFrame, save: bool = True):
    """2D PCA scatter coloured by KMeans cluster label."""
    fig = px.scatter(
        pca_df, x="PC1", y="PC2",
        color="cluster",
        hover_data=pca_df.columns.tolist(),
        title="Trader Profiles — PCA + KMeans Clustering",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        template=PLOTLY_TEMPLATE,
        opacity=0.7,
    )
    if save:
        path = FIG_DIR / "06_pca_clusters.html"
        fig.write_html(str(path))
        print(f"[📊] Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame, save: bool = True):
    """Horizontal bar chart of XGBoost/LightGBM feature importance."""
    top = importance_df.nlargest(20, "importance")

    fig = px.bar(
        top, x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        title="Feature Importance — PnL Direction Prediction",
        labels={"importance": "Importance Score", "feature": "Feature"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    if save:
        path = FIG_DIR / "07_feature_importance.html"
        fig.write_html(str(path))
        print(f"[📊] Saved → {path}")
    return fig
