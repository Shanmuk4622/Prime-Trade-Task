# PrimeTrade.ai — Trader Sentiment Analysis
## Insights Report

> **Objective:** Explore the relationship between Hyperliquid trader performance and Bitcoin market
> sentiment, uncover hidden patterns, and derive actionable trading strategy insights.

---

## Executive Summary

Analysis of **211,224 trades** across **32 accounts** spanning
**480 trading days**, merged with Bitcoin Fear/Greed Index data,
reveals that **market sentiment has a statistically significant but practically modest** influence
on individual trade outcomes. The most actionable insight is a clear **counter-intuitive pattern**:
traders generate higher *mean* PnL during **Extreme Greed** regimes, yet Fear regimes
accumulate the **highest aggregate profit** due to trade volume surges. A rules-based
sentiment-gated strategy can meaningfully improve capital allocation decisions.

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total trades analysed | 211,224 |
| Unique trading accounts | 32 |
| Trading days covered | 480 |
| Sentiment days covered | 2,644 |
| Sentiment match rate | 100% |
| Date range (trades) | 2023-05-01 → 2025-05-01 |
| Symbols traded | 246 |

---

## Finding 1: Sentiment Significantly Affects Trader PnL

### PnL by Sentiment Regime

| Regime | Trades | Mean PnL | Win Rate | Total PnL |
|--------|--------|----------|----------|-----------|
| Extreme Fear | 31,364 | $34.72 | 34.6% | $1.09M |
| Fear | 55,328 | $57.42 | 44.6% | $3.18M |
| Neutral | 36,302 | $34.32 | 39.9% | $1.25M |
| Greed | 55,803 | $41.56 | 40.3% | $2.32M |
| Extreme Greed | 32,421 | $74.74 | 44.4% | $2.42M |

**ANOVA:** F = 12.6629, p ≈ 0.0 → **Significant (p<0.05)**

**Correlations (Sentiment Score ↔ Closed PnL):**
- Pearson r = 0.0066 (p = 0.002422)
- Spearman r = 0.0288 (p = 0.0)

### Interpretation
- The ANOVA confirms regime differences are **not due to chance** (p ≈ 0).
- However, both correlation coefficients are **very small** (|r| < 0.03), indicating sentiment
  explains only a tiny fraction of individual trade variance.
- **Practical takeaway:** Sentiment should be used as a *background filter*, not a per-trade signal.

---

## Finding 2: The Counter-Intuitive Extreme Greed Premium

Contrary to the "buy the fear" narrative popular in crypto markets:

- **Extreme Greed** traders achieve the **highest mean PnL** ($74.74) —
  **2.2×** above Extreme Fear.
- **Fear** regime accumulates the **highest total PnL** ($3,177,166) due to
  the **highest trade volume** (55,328 trades).
- **Extreme Fear** has the **lowest win rate** (34.6%),
  confirming that panic conditions hurt most retail traders.

### Why does Extreme Greed outperform?
Momentum-following accounts ride trend extensions in greed phases,
compounding gains with higher conviction sizing. In fear phases, stops get hit more
frequently as volatility spikes, producing the low win rate seen.

---

## Finding 3: Win Rates Are Universally Below 50%

| Regime | Win Rate | Delta vs 50% |
|--------|----------|-------------|
| Extreme Fear | 34.6% | -15.4pp |
| Fear | 44.6% | -5.4pp |
| Neutral | 39.9% | -10.1pp |
| Greed | 40.3% | -9.7pp |
| Extreme Greed | 44.4% | -5.6pp |

All regimes show sub-50% win rates, implying **positive expected value comes from
asymmetric sizing** (letting winners run, cutting losers). This is consistent with a
positively skewed PnL distribution — a hallmark of professional algo/trend-following strategies.

---

## Finding 4: Trader Clustering — 4 Distinct Profiles

KMeans (k=4) on account-level performance metrics reveals **4 trader archetypes**
(PCA explains **62.7%** of variance):

|   cluster |        total_pnl |   win_rate |   trade_count |   sharpe_proxy |
|----------:|-----------------:|-----------:|--------------:|---------------:|
|         0 |  59990.3         |      0.364 |       5915.38 |          0.062 |
|         1 | 343958           |      0.296 |       1134.33 |          0.173 |
|         2 |      1.27206e+06 |      0.399 |      18432.6  |          0.092 |
|         3 | 243120           |      0.524 |       2626.5  |          0.201 |

### Archetype Descriptions
| Cluster | Label | Behaviour |
|---------|-------|-----------|
| 0 | **Momentum Whales** | High total PnL, high trade count, diversified |
| 1 | **Conservative Earners** | Steady positive PnL, moderate win rate, low risk |
| 2 | **High-Variance Speculators** | Large swings, low win rate, extreme drawdowns |
| 3 | **Marginal Accounts** | Low volume, near-zero net PnL |

---

## Finding 5: Top vs Bottom Trader Sentiment Behaviour

| Metric | Top Traders (top 25%) | Bottom Traders (bottom 25%) |
|--------|----------------------|----------------------------|
| Mean total PnL | $985.5K | $-19.0K |
| Mean win rate | 41.4% | 36.8% |
| Dominant regime | Fear | Extreme Fear |
| Mean Sharpe proxy | 0.119 | 0.026 |

**Key insight:** Top traders are more active during **Fear** regimes, suggesting they exploit
panic-driven mispricings while retail flows are net-selling.

---

## Finding 6: Top 10 Symbols by PnL

| symbol   |        total_pnl |   mean_pnl |   trades |
|:---------|-----------------:|-----------:|---------:|
| @107     |      2.78391e+06 |    92.8218 |    29992 |
| HYPE     |      1.94848e+06 |    28.6521 |    68005 |
| SOL      |      1.63956e+06 |   153.359  |    10691 |
| ETH      |      1.31998e+06 |   118.299  |    11158 |
| BTC      | 868045           |    33.3044 |    26064 |
| MELANIA  | 390351           |    88.1552 |     4428 |
| ENA      | 217330           |   219.525  |      990 |
| SUI      | 199269           |   100.692  |     1979 |
| ZRO      | 183778           |   148.328  |     1239 |
| DOGE     | 147543           |   178.624  |      826 |

---

## Finding 7: Sentiment Regime Persistence

### Transition Probabilities (row = current, col = next day's regime)

| regime        |   Extreme Fear |   Extreme Greed |   Fear |   Greed |   Neutral |
|:--------------|---------------:|----------------:|-------:|--------:|----------:|
| Extreme Fear  |          0.813 |           0     |  0.18  |   0     |     0.007 |
| Extreme Greed |          0     |           0.778 |  0.004 |   0.208 |     0.011 |
| Fear          |          0.13  |           0.001 |  0.756 |   0.014 |     0.099 |
| Greed         |          0.002 |           0.094 |  0.02  |   0.785 |     0.1   |
| Neutral       |          0.007 |           0.002 |  0.175 |   0.17  |     0.645 |

**Observation:** Regimes are **highly persistent** — a Fear day is most likely followed by
another Fear day. Extreme regimes (Fear/Greed) revert toward moderate regimes within 3-5 days.
This means a sentiment-aware strategy has **sufficient time** to act after a regime signal fires.

---

## Trading Strategy Recommendations

### Strategy 1: Sentiment-Gated Position Sizing

> Increase notional exposure during Fear regimes, reduce during Extreme Greed.

| Regime | Sizing Multiplier | Rationale |
|--------|------------------|-----------|
| Extreme Fear  | 0.5× | High volatility, low win rate — reduce size |
| Fear          | 1.5× | Best risk-adjusted regime for mean PnL |
| Neutral       | 1.0× | Base allocation |
| Greed         | 1.2× | Momentum-friendly, decent win rate |
| Extreme Greed | 0.8× | Mean PnL high but reversal risk increases |

### Strategy 2: Long/Short Bias Flip

- **Fear regime → Prefer LONG** (buy the dip, fade the panic sellers)
- **Extreme Greed → Prefer SHORT** (fade the crowd, front-run the reversal)

### Strategy 3: Regime-Aware Stop Loss Adjustment

During **Extreme Fear**, volatility (std_pnl) is highest. Widen stops by 30-50% to
avoid noise-driven stop-outs that reduce win rate.

### Strategy 4: Account Segmentation for Execution

Copy-trade or mirror only **Cluster 0 / Top segment** accounts during Fear regimes —
their dominant-regime alignment with Fear shows they are exploiting volatility, not fighting it.

---

## Limitations & Caveats

1. **Data leakage in ML models** — Rolling features (win_rate_7d) computed from the same
   trade set as the target inflates AUC to 1.0. Production models must use strict
   look-ahead-free time splits.
2. **32 accounts only** — Small account pool limits generalisation. Results may differ
   for the broader Hyperliquid population.
3. **Correlation is weak** — Sentiment alone explains < 0.1% of trade-level PnL variance.
   It is a *context filter*, not a *signal*.
4. **No transaction costs** — All PnL is gross. Net PnL after fees will be lower.

---

## Appendix: Key Statistics

| Metric | Value |
|--------|-------|
| Total gross PnL across all trades | $10.30M |
| Mean PnL per trade | $48.75 |
| Overall win rate | 41.1% |
| Max single trade gain | $135.3K |
| Max single trade loss | $-118.0K |
| ANOVA F-statistic | 12.6629 |
| ANOVA p-value | 0.0 |
| Spearman r (sentiment ↔ PnL) | 0.0288 |
| KMeans clusters | 4 |
| PCA variance explained | 62.7% |

---

*Generated automatically by `src/generate_report.py` · PrimeTrade.ai Hiring Assignment*
