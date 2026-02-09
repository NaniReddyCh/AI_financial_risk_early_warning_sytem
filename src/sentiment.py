import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class NewsSentimentScorer:
    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def score_headlines(self, news: pd.DataFrame) -> pd.DataFrame:
        scored = news.copy()
        scored["sentiment"] = scored["headline"].apply(self._score_text)
        return scored

    def _score_text(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        return self._analyzer.polarity_scores(text)["compound"]


def aggregate_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    grouped = news.groupby("date").agg(
        sentiment_mean=("sentiment", "mean"),
        sentiment_min=("sentiment", "min"),
        sentiment_max=("sentiment", "max"),
        news_count=("sentiment", "size"),
    )
    grouped = grouped.reset_index()
    grouped["negative_headline_share"] = (
        news.groupby("date")["sentiment"].apply(lambda series: (series < -0.2).mean())
    ).values
    return grouped
