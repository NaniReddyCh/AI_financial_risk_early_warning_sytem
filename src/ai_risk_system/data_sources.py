from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance multi-index columns into single-level names."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def fetch_market_data(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch market data from Yahoo Finance and normalize column names."""
    import yfinance as yf

    hist = yf.Ticker(symbol).history(period=period, interval=interval)
    if hist.empty:
        raise ValueError(f"No market data returned for {symbol!r}.")

    hist = hist.reset_index()
    hist = _flatten_yfinance_columns(hist)

    rename_map = {
        "Date": "date",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    hist = hist.rename(columns=rename_map)

    keep_cols = [col for col in ["date", "close", "adj_close", "volume"] if col in hist.columns]
    required_price_cols = [col for col in ["close", "adj_close"] if col in hist.columns]
    if not required_price_cols:
        raise ValueError("Yahoo response missing both close and adj_close columns.")

    return hist[keep_cols].dropna(subset=required_price_cols).copy()


def parse_news_feed(rows: Iterable[dict]) -> tuple[datetime, list[str]]:
    """Normalizes an API/news feed payload into a day+headline list."""
    headlines = [str(r.get("title", "")).strip() for r in rows if r.get("title")]
    return datetime.utcnow(), headlines
