"""
preprocessing.py
================
Clean, type-cast, and deduplicate the raw datasets.
Outputs go to data/processed/.
"""

import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Sentiment regime labels ────────────────────────────────────────────────────
REGIME_BINS   = [0, 25, 45, 55, 75, 100]
REGIME_LABELS = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Hyperliquid trades dataframe.

    Steps
    -----
    1. Parse timestamps → datetime (UTC)
    2. Cast numeric columns
    3. Drop full duplicates
    4. Remove rows with null PnL (unclosed positions)
    5. Add a 'trade_date' column (date only, for join)

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_trades()

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()
    # Normalise column names (idempotent — safe if already normalised)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── 1. Timestamps ──────────────────────────────────────────────────────────
    # Real CSV has: 'timestamp_ist' (string dd-mm-yyyy HH:MM) and 'timestamp' (epoch seconds)
    # Parse the human-readable IST column first; fall back to epoch
    if "timestamp_ist" in df.columns:
        df["timestamp_ist"] = pd.to_datetime(
            df["timestamp_ist"], format="%d-%m-%Y %H:%M", errors="coerce"
        )
        df["trade_date"] = df["timestamp_ist"].dt.normalize()
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["trade_date"] = df["timestamp"].dt.normalize()

    # ── 2. Standardise column aliases ─────────────────────────────────────────
    # Rename 'coin' → 'symbol' for downstream consistency
    if "coin" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"coin": "symbol"})

    # ── 3. Numeric casting ─────────────────────────────────────────────────────
    # Real columns after normalisation: closed_pnl, size_tokens, size_usd, execution_price, fee
    numeric_candidates = [
        "closed_pnl", "size_tokens", "size_usd",
        "execution_price", "start_position", "fee",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 4. Deduplicate ─────────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"  [!] Dropped {before - after:,} duplicate trade rows")

    # ── 5. Drop null PnL ───────────────────────────────────────────────────────
    pnl_col = "closed_pnl" if "closed_pnl" in df.columns else None
    if pnl_col:
        null_count = df[pnl_col].isna().sum()
        df = df.dropna(subset=[pnl_col])
        if null_count:
            print(f"  [!] Dropped {null_count:,} rows with null closed_pnl")

    # ── 6. Side normalisation ──────────────────────────────────────────────────
    if "side" in df.columns:
        df["side"] = df["side"].str.strip().str.upper()  # → 'BUY' / 'SELL'

    print(f"[✓] Trades cleaned      → {len(df):,} rows remain")
    return df.reset_index(drop=True)


def clean_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Fear/Greed Index dataframe.

    Steps
    -----
    1. Parse 'date' column
    2. Cast 'value' to numeric (if present)
    3. Map classification to ordered categorical
    4. Add 'regime' bin column from numeric value
    5. Deduplicate by date

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_sentiment()

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 'trade_date' as date key.
    """
    df = df.copy()

    # ── 1. Parse date ──────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["trade_date"] = df["date"].dt.normalize()

    # ── 2. Numeric value ───────────────────────────────────────────────────────
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # Build regime from numeric score
        df["regime"] = pd.cut(df["value"], bins=REGIME_BINS, labels=REGIME_LABELS, right=True)
    else:
        # Fallback: derive regime from classification string
        classification_map = {
            "extreme fear": "Extreme Fear",
            "fear": "Fear",
            "neutral": "Neutral",
            "greed": "Greed",
            "extreme greed": "Extreme Greed",
        }
        df["regime"] = df["classification"].str.lower().map(classification_map)

    # ── 3. Ordered categorical ─────────────────────────────────────────────────
    cat_type = pd.CategoricalDtype(categories=REGIME_LABELS, ordered=True)
    df["regime"] = df["regime"].astype(cat_type)

    # ── 4. Deduplicate by date ─────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["trade_date"])
    after = len(df)
    if before != after:
        print(f"  [!] Dropped {before - after:,} duplicate sentiment rows")

    print(f"[✓] Sentiment cleaned   → {len(df):,} rows remain")
    return df.sort_values("trade_date").reset_index(drop=True)


def save_processed(df: pd.DataFrame, name: str) -> pathlib.Path:
    """Save a processed dataframe to data/processed/<name>.csv."""
    out_path = PROCESSED_DIR / name
    df.to_csv(out_path, index=False)
    print(f"[💾] Saved → {out_path}")
    return out_path
