from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd


def fetch_market_data(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch market data from Yahoo Finance and normalize column names."""
    import yfinance as yf

    hist = yf.Ticker(symbol).history(period=period, interval=interval)
    if hist.empty:
        raise ValueError(f"No market data returned for {symbol!r}.")

    hist = hist.reset_index()
    rename_map = {
        "Date": "date",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    hist = hist.rename(columns=rename_map)

    keep_cols = [col for col in ["date", "close", "adj_close", "volume"] if col in hist.columns]
    return hist[keep_cols].dropna(subset=[col for col in ["close", "adj_close"] if col in hist.columns]).copy()


def parse_news_feed(rows: Iterable[dict]) -> tuple[datetime, list[str]]:
    """Normalizes an API/news feed payload into a day+headline list."""
    headlines = [str(r.get("title", "")).strip() for r in rows if r.get("title")]
    return datetime.utcnow(), headlines
