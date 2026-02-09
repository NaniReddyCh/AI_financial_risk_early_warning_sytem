# Data Sources (Download Manually)

This project does **not** ship datasets. Download the datasets below and save them as:

- `data/market.csv`
- `data/news.csv`

## Market Data (OHLCV)
Choose one of the following:

- **Kaggle: Historical Stock Prices** (daily OHLCV). https://www.kaggle.com/datasets/ehallmar/stocks-historical-price-data
- **Kaggle: S&P 500 Stock Data** (daily prices). https://www.kaggle.com/datasets/camnugent/sandp500
- **Stooq daily prices** (CSV downloads). https://stooq.com/db/h/

**Expected columns:** `date, open, high, low, close, volume`

## News / Headlines Data
Choose one of the following:

- **Kaggle: Financial News Headlines** (headline + date). https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
- **Kaggle: Reuters News Headlines** (date + title). https://www.kaggle.com/datasets/financedata/reuters-news-headlines
- **GDELT** (global news with tone). https://www.gdeltproject.org/

**Expected columns:** `date, headline`

## Notes
- If your dataset uses different column names, rename them to match the expected schema.
- The pipeline assumes daily frequency and merges by date.
