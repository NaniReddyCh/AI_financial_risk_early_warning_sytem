# AI Financial Risk Early Warning System

An end-to-end, runnable baseline for detecting early financial risk signals using market data and news sentiment. This repository includes data loaders, feature engineering, sentiment scoring, a risk labeling strategy, model training, and a simple dashboard.

## What This Repository Delivers
- Market + news ingestion with strict schemas.
- Technical indicators and daily news sentiment aggregation.
- Risk labeling based on forward price drops.
- Baseline logistic regression model with evaluation metrics.
- Exported model, metrics, and risk probability predictions.
- Streamlit dashboard for quick visualization.

## Quickstart

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Download datasets
Download the datasets listed in `data/README.md` and save them as:
- `data/market.csv`
- `data/news.csv`

### 3) Run the training pipeline
```bash
python -m src.run_pipeline --market data/market.csv --news data/news.csv --output reports
```

Artifacts will be written to:
- `reports/risk_model.joblib`
- `reports/metrics.json`
- `reports/risk_predictions.csv`

### 4) Launch the dashboard
```bash
streamlit run src/dashboard.py -- --predictions reports/risk_predictions.csv
```

## Expected Dataset Schema

### Market data (`data/market.csv`)
Columns required:
- `date` (YYYY-MM-DD)
- `open`, `high`, `low`, `close`
- `volume`

### News data (`data/news.csv`)
Columns required:
- `date` (YYYY-MM-DD)
- `headline`

If your dataset uses other column names, rename them to match the schema.

## Repository Structure
```
.
├── data/               # local data downloads (not committed)
├── models/             # saved models (optional)
├── reports/            # metrics and predictions
├── src/                # pipeline + dashboard code
├── notebooks/          # optional experiments
└── requirements.txt    # dependencies
```

## How the Risk Label Works
A row is labeled as **high risk** if the future close (N days ahead) drops by more than a threshold (e.g., -5%). Adjust these parameters with:
- `--horizon` (default 10 days)
- `--threshold` (default -0.05)

## Next Improvements (Optional)
- Swap logistic regression for gradient boosting (XGBoost/LightGBM).
- Add macro indicators (FRED) or volatility indices.
- Use a finance-specific transformer (FinBERT) for richer sentiment.
