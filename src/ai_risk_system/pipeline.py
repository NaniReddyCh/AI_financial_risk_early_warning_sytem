from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from .sentiment import FinancialSentimentAnalyzer


@dataclass
class PipelineConfig:
    """Configuration for training and scoring."""

    lookahead_days: int = 7
    drop_threshold: float = 0.05
    market_weight: float = 0.7
    sentiment_weight: float = 0.3


class FinancialRiskPipeline:
    """Train a market risk model and combine it with pretrained sentiment inference."""

    feature_columns = ["return_1d", "return_5d", "volatility_10d", "volume_zscore", "sma_ratio"]

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = LogisticRegression(max_iter=500)
        self.sentiment = FinancialSentimentAnalyzer()

    def build_features(self, market_df: pd.DataFrame) -> pd.DataFrame:
        df = market_df.copy().sort_values("date")
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["volatility_10d"] = df["close"].pct_change().rolling(10).std()
        df["volume_zscore"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
        df["sma_ratio"] = df["close"] / df["close"].rolling(20).mean()

        future_return = df["close"].shift(-self.config.lookahead_days) / df["close"] - 1
        df["target"] = (future_return <= -self.config.drop_threshold).astype(int)
        return df.dropna().reset_index(drop=True)

    def aggregate_sentiment(self, headlines: Iterable[str]) -> float:
        scores = [self.sentiment.score_headline(text) for text in headlines if text]
        if not scores:
            return 0.5
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def train(self, market_df: pd.DataFrame) -> dict:
        df = self.build_features(market_df)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        x_train = train_df[self.feature_columns]
        y_train = train_df["target"]
        x_test = test_df[self.feature_columns]
        y_test = test_df["target"]

        self.model.fit(x_train, y_train)
        prob = self.model.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        return {
            "classification_report": classification_report(y_test, pred, output_dict=True),
            "roc_auc": float(roc_auc_score(y_test, prob)) if y_test.nunique() > 1 else 0.5,
        }

    def score_latest(self, latest_market_window: pd.DataFrame, headlines: Iterable[str]) -> dict:
        features_df = self.build_features(latest_market_window)
        if features_df.empty:
            raise ValueError("Not enough market history to build indicators.")

        latest_features = features_df[self.feature_columns].iloc[[-1]]
        market_risk_score = float(self.model.predict_proba(latest_features)[0, 1])
        sentiment_score = self.aggregate_sentiment(headlines)

        final_score = (
            self.config.market_weight * market_risk_score
            + self.config.sentiment_weight * sentiment_score
        )
        if final_score >= 0.7:
            category = "High"
        elif final_score >= 0.4:
            category = "Medium"
        else:
            category = "Low"

        return {
            "market_risk_score": market_risk_score,
            "sentiment_score": sentiment_score,
            "final_risk_score": float(final_score),
            "risk_category": category,
        }

    def save_model(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "config": self.config,
                "feature_columns": self.feature_columns,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: Path) -> "FinancialRiskPipeline":
        import joblib

        bundle = joblib.load(path)
        pipeline = cls(bundle["config"])
        pipeline.model = bundle["model"]
        return pipeline
