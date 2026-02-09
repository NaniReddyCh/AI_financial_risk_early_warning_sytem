import pandas as pd


def add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["return_1d"] = data["close"].pct_change()
    data["return_5d"] = data["close"].pct_change(5)
    data["sma_10"] = data["close"].rolling(window=10).mean()
    data["sma_20"] = data["close"].rolling(window=20).mean()
    data["ema_10"] = data["close"].ewm(span=10, adjust=False).mean()
    data["volatility_10"] = data["return_1d"].rolling(window=10).std()
    data["volatility_20"] = data["return_1d"].rolling(window=20).std()

    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["rsi_14"] = 100 - (100 / (1 + rs))
    return data
