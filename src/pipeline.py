import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data_loaders import load_market_data, load_news_data
from src.indicators import add_technical_indicators
from src.sentiment import NewsSentimentScorer, aggregate_daily_sentiment


@dataclass
class PipelineConfig:
    risk_horizon_days: int = 10
    risk_drop_threshold: float = -0.05
    test_split_ratio: float = 0.2
    random_state: int = 42


def build_labels(data: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    future_close = data["close"].shift(-horizon)
    future_return = (future_close / data["close"]) - 1.0
    return (future_return <= threshold).astype(int)


def prepare_dataset(market_path: str, news_path: str) -> pd.DataFrame:
    market = load_market_data(market_path)
    market = add_technical_indicators(market)

    news = load_news_data(news_path)
    scorer = NewsSentimentScorer()
    scored_news = scorer.score_headlines(news)
    sentiment = aggregate_daily_sentiment(scored_news)

    merged = pd.merge(market, sentiment, on="date", how="left")
    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0.0)
    merged["sentiment_min"] = merged["sentiment_min"].fillna(0.0)
    merged["sentiment_max"] = merged["sentiment_max"].fillna(0.0)
    merged["news_count"] = merged["news_count"].fillna(0.0)
    merged["negative_headline_share"] = merged["negative_headline_share"].fillna(0.0)
    return merged.dropna().reset_index(drop=True)


def split_train_test(data: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(data) * (1 - test_ratio))
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    return train, test


def train_model(train: pd.DataFrame, test: pd.DataFrame, config: PipelineConfig) -> dict:
    features = [
        "return_1d",
        "return_5d",
        "sma_10",
        "sma_20",
        "ema_10",
        "volatility_10",
        "volatility_20",
        "rsi_14",
        "sentiment_mean",
        "sentiment_min",
        "sentiment_max",
        "news_count",
        "negative_headline_share",
    ]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features])
    x_test = scaler.transform(test[features])
    y_train = train["risk_label"]
    y_test = test["risk_label"]

    model = LogisticRegression(max_iter=200, random_state=config.random_state)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    report = classification_report(y_test, predictions, output_dict=True)
    report["roc_auc"] = roc_auc_score(y_test, probabilities)

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "report": report,
        "predictions": predictions,
        "probabilities": probabilities,
        "y_test": y_test,
        "test_index": test.index,
    }


def run_pipeline(market_path: str, news_path: str, output_dir: str, config: PipelineConfig) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(market_path, news_path)
    dataset["risk_label"] = build_labels(
        dataset, horizon=config.risk_horizon_days, threshold=config.risk_drop_threshold
    )

    train, test = split_train_test(dataset, test_ratio=config.test_split_ratio)
    results = train_model(train, test, config=config)

    model_bundle = {
        "model": results["model"],
        "scaler": results["scaler"],
        "features": results["features"],
        "config": config,
    }
    joblib.dump(model_bundle, output_path / "risk_model.joblib")

    metrics_path = output_path / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results["report"], handle, indent=2)

    predictions_frame = test[["date", "close", "risk_label"]].copy()
    predictions_frame["risk_probability"] = results["probabilities"]
    predictions_frame.to_csv(output_path / "risk_predictions.csv", index=False)
