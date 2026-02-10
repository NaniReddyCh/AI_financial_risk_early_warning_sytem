# AI Financial Risk Early Warning System

> **Final confirmation:** You train the market-risk model yourself, while sentiment is produced by a pretrained financial NLP model.

This project implements a practical **early warning pipeline** for financial instability prediction. It predicts a **risk score (0–1)** and **risk category (Low/Medium/High)** using:

- A trained market model (technical + volatility indicators).
- A pretrained sentiment signal from financial news.
- Weighted fusion: `0.7 * market_risk + 0.3 * sentiment`.

## Architecture

### Part A — Trained Market Model
- Source: Yahoo Finance historical OHLCV data.
- Features: 1D/5D return, 10D volatility, volume z-score, SMA ratio.
- Label: high risk if future return over a 7-day horizon is <= -5%.
- Model: logistic regression baseline (easy to swap with XGBoost/LightGBM).

### Part B — Pretrained Sentiment Inference
- Primary mode: FinBERT (`ProsusAI/finbert`) via `transformers` if installed.
- Fallback mode: lightweight financial keyword lexicon.

### Fusion and Output
- `final_risk_score = 0.7 * market_risk_score + 0.3 * sentiment_score`
- Category thresholds:
  - `>= 0.7`: High
  - `>= 0.4`: Medium
  - else: Low

## Project Structure

```text
src/ai_risk_system/
  cli.py             # train + daily score commands
  data_sources.py    # Yahoo data fetch + news payload parser
  pipeline.py        # feature engineering, label creation, training, scoring
  sentiment.py       # pretrained sentiment inference + fallback
tests/
  test_pipeline.py   # synthetic-data test of train/score flow
```

## Quick Start

```bash
python -m pip install -e .
```

Optional pretrained NLP support:

```bash
python -m pip install -e .[nlp]
```

### 1) Train market model

```bash
risk-ews train --symbol ^GSPC --period 5y --model-path artifacts/risk_model.joblib
```

### 2) Run daily scoring

```bash
risk-ews score --symbol ^GSPC --period 1y --model-path artifacts/risk_model.joblib \
  --headline "Bank warns of recession risks" \
  --headline "Large-cap tech beats estimates"
```

Returns JSON like:

```json
{
  "market_risk_score": 0.48,
  "sentiment_score": 0.36,
  "final_risk_score": 0.444,
  "risk_category": "Medium"
}
```

## Evaluation

- ML metrics (from training split): precision, recall, F1, ROC-AUC.
- Business lens (to add in notebooks/dashboard):
  - early-warning timeliness before drawdowns,
  - max drawdown reduction,
  - Sharpe improvement.

## Next Best Upgrades

1. Replace logistic regression with LightGBM/XGBoost.
2. Add a dashboard (Streamlit) for trend view and alerts.
3. Add scheduled daily batch job (cron/Airflow/GitHub Actions).
4. Add event backtests across stress periods (COVID crash, sector corrections).
