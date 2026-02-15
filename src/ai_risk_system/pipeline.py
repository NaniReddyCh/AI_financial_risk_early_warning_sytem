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
    drop_threshold: float = 0.03
    market_weight: float = 0.7
    sentiment_weight: float = 0.3


class FinancialRiskPipeline:
    """Market-risk pipeline with external pretrained model support."""

    feature_columns = [
        "ret_1d",
        "ret_5d",
        "volatility_14",
        "ma20",
        "ma50",
        "price_ma_ratio",
    ]

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        # Placeholder for optional local experiments; production scoring should load a pretrained model.
        self.model = LogisticRegression(max_iter=500)
        self.sentiment = FinancialSentimentAnalyzer()

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        out.columns = [str(c).strip() for c in out.columns]
        return out

    @staticmethod
    def _price_column(df: pd.DataFrame) -> str:
        for col in ("adj_close", "Adj Close", "close", "Close"):
            if col in df.columns:
                return col
        raise ValueError("Market dataframe must include one of: adj_close, Adj Close, close, Close")

    def build_features(self, market_df: pd.DataFrame, include_label: bool = False) -> pd.DataFrame:
        df = self._flatten_columns(market_df)
        if "date" in df.columns:
            df = df.sort_values("date")

        price_col = self._price_column(df)
        df["ret_1d"] = df[price_col].pct_change()
        df["ret_5d"] = df[price_col].pct_change(5)
        df["volatility_14"] = df["ret_1d"].rolling(14).std()
        df["ma20"] = df[price_col].rolling(20).mean()
        df["ma50"] = df[price_col].rolling(50).mean()
        df["price_ma_ratio"] = df[price_col] / df["ma20"]

        if include_label:
            future_return = df[price_col].shift(-self.config.lookahead_days) / df[price_col] - 1
            df["risk_label"] = (future_return <= -self.config.drop_threshold).astype(int)

        return df.dropna().reset_index(drop=True)

    def aggregate_sentiment(self, headlines: Iterable[str]) -> float:
        scores = [self.sentiment.score_headline(text) for text in headlines if text]
        if not scores:
            return 0.5
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def train(self, market_df: pd.DataFrame) -> dict:
        """Optional helper for local experiments. In production, load external pretrained model."""
        df = self.build_features(market_df, include_label=True)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        x_train = train_df[self.feature_columns]
        y_train = train_df["risk_label"]
        x_test = test_df[self.feature_columns]
        y_test = test_df["risk_label"]

        self.model.fit(x_train, y_train)
        prob = self.model.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        return {
            "classification_report": classification_report(y_test, pred, output_dict=True),
            "roc_auc": float(roc_auc_score(y_test, prob)) if y_test.nunique() > 1 else 0.5,
        }

    def score_latest(self, latest_market_window: pd.DataFrame, headlines: Iterable[str]) -> dict:
        features_df = self.build_features(latest_market_window, include_label=False)
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
        """Load either a bundled model dict or a raw sklearn model object."""
        import joblib

        artifact = joblib.load(path)
        if isinstance(artifact, dict) and "model" in artifact:
            pipeline = cls(artifact.get("config"))
            pipeline.model = artifact["model"]
            if artifact.get("feature_columns"):
                pipeline.feature_columns = artifact["feature_columns"]
            return pipeline

        pipeline = cls()
        pipeline.model = artifact
        return pipeline
