from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd


def fetch_market_data(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV market data from Yahoo Finance."""
    import yfinance as yf

    hist = yf.Ticker(symbol).history(period=period, interval=interval)
    if hist.empty:
        raise ValueError(f"No market data returned for {symbol!r}.")
    hist = hist.reset_index()
    hist = hist.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
    return hist[["date", "close", "volume"]].dropna()


def parse_news_feed(rows: Iterable[dict]) -> tuple[datetime, list[str]]:
    """Normalizes an API/news feed payload into a day+headline list."""
    headlines = [str(r.get("title", "")).strip() for r in rows if r.get("title")]
    return datetime.utcnow(), headlines
