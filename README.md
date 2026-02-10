# AI Financial Risk Early Warning System

> **Final confirmation:** You train the market-risk model yourself, while sentiment is produced by a pretrained financial NLP model.

This project implements an **early warning pipeline** for financial instability prediction. It outputs:
- `final_risk_score` (0 to 1)
- `risk_category` (`Low`, `Medium`, `High`)

## Key behavior (updated)

- ✅ **Market model is expected to be pretrained externally** and loaded from disk for scoring.
- ✅ News sentiment uses a pretrained NLP model (FinBERT when available, fallback lexicon otherwise).
- ✅ Daily scoring does **not retrain** the market model.

## Market Feature Contract

The market model uses these features:

```python
features = [
    "ret_1d",
    "ret_5d",
    "volatility_14",
    "ma20",
    "ma50",
    "price_ma_ratio",
]
```

Feature engineering logic:

```python
price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
df["ret_1d"] = df[price_col].pct_change()
df["ret_5d"] = df[price_col].pct_change(5)
df["volatility_14"] = df["ret_1d"].rolling(14).std()
df["ma20"] = df[price_col].rolling(20).mean()
df["ma50"] = df[price_col].rolling(50).mean()
df["price_ma_ratio"] = df[price_col] / df["ma20"]
```

In this codebase, equivalent support is implemented for normalized Yahoo columns (`adj_close` / `close`).

## Fusion Rule

```text
final_risk_score = 0.7 * market_risk_score + 0.3 * sentiment_score
```

## Streamlit Dashboard

Run:

```bash
streamlit run src/ai_risk_system/dashboard.py
```

or:

```bash
risk-ews-dashboard
```

Dashboard behavior:
- Loads your pretrained market model path
- Fetches latest Yahoo market data
- Builds required features
- Runs risk scoring (no market retraining)
- Displays clean card + charts + headlines

## CLI

### Score using pretrained market model (recommended)

```bash
risk-ews score --symbol ^GSPC --period 1y --model-path artifacts/risk_model.joblib \
  --headline "Bank warns of recession risks" \
  --headline "Large-cap tech beats estimates"
```

### Optional local training helper

```bash
risk-ews train --symbol ^GSPC --period 5y --model-path artifacts/risk_model.joblib
```

Use this only for local experimentation; production flow should use your external pretrained market model.
