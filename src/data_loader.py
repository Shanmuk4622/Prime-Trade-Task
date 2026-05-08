"""
data_loader.py
==============
Load and validate the two raw datasets:
  • historical_trades.csv  — Hyperliquid trader data
  • fear_greed_index.csv   — Bitcoin market sentiment
"""

import pathlib
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

TRADES_FILE   = RAW_DIR / "historical_data.csv"
SENTIMENT_FILE = RAW_DIR / "fear_greed_index.csv"

# ── Expected column sets (loose validation) ────────────────────────────────────
# After .str.lower().str.replace(" ", "_") normalisation:
TRADES_REQUIRED_COLS   = {"account", "coin", "side", "closed_pnl"}
SENTIMENT_REQUIRED_COLS = {"date", "classification"}


def load_trades(path: pathlib.Path = TRADES_FILE) -> pd.DataFrame:
    """
    Load Hyperliquid historical trader CSV.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns intact.

    Raises
    ------
    FileNotFoundError
        If the CSV is not present in data/raw/.
    ValueError
        If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Trades file not found at: {path}\n"
            "Download it from the Google Drive link and place it in data/raw/."
        )

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    missing = TRADES_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Trades CSV is missing required columns: {missing}")

    print(f"[✓] Trades loaded       → {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def load_sentiment(path: pathlib.Path = SENTIMENT_FILE) -> pd.DataFrame:
    """
    Load Bitcoin Fear/Greed Index CSV.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns intact.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Sentiment file not found at: {path}\n"
            "Download it from the Google Drive link and place it in data/raw/."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Handle both possible column name spellings
    if "classification" not in df.columns and "sentiment" in df.columns:
        df = df.rename(columns={"sentiment": "classification"})

    missing = SENTIMENT_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Sentiment CSV is missing required columns: {missing}")

    print(f"[✓] Sentiment loaded    → {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def load_both() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper — returns (trades_df, sentiment_df)."""
    return load_trades(), load_sentiment()
