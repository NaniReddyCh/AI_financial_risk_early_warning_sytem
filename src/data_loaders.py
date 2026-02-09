import pandas as pd


REQUIRED_MARKET_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
REQUIRED_NEWS_COLUMNS = {"date", "headline"}


def load_market_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = REQUIRED_MARKET_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Market data missing columns: {sorted(missing)}")
    data["date"] = pd.to_datetime(data["date"]).dt.date
    data = data.sort_values("date").reset_index(drop=True)
    return data


def load_news_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = REQUIRED_NEWS_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"News data missing columns: {sorted(missing)}")
    data["date"] = pd.to_datetime(data["date"]).dt.date
    data["headline"] = data["headline"].fillna("")
    return data
