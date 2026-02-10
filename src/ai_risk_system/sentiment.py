from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _KeywordLexicon:
    positive: tuple[str, ...] = (
        "growth",
        "record profit",
        "beats estimates",
        "upgrade",
        "bullish",
        "rally",
    )
    negative: tuple[str, ...] = (
        "downgrade",
        "misses estimates",
        "lawsuit",
        "recession",
        "bankruptcy",
        "selloff",
        "crash",
    )


class FinancialSentimentAnalyzer:
    """Uses a pretrained model when available; fallback to a lightweight lexicon."""

    def __init__(self) -> None:
        self._lexicon = _KeywordLexicon()
        self._pipeline = None
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
            )
        except Exception:
            # Fallback allows the project to run without heavyweight NLP dependencies.
            self._pipeline = None

    def score_headline(self, text: str) -> float:
        text = (text or "").lower().strip()
        if not text:
            return 0.5

        if self._pipeline is not None:
            try:
                result = self._pipeline(text, truncation=True)[0]
                label = str(result["label"]).lower()
                confidence = float(result["score"])
                if "positive" in label:
                    return 0.5 + 0.5 * confidence
                if "negative" in label:
                    return 0.5 - 0.5 * confidence
                return 0.5
            except Exception:
                pass

        positive_hits = sum(token in text for token in self._lexicon.positive)
        negative_hits = sum(token in text for token in self._lexicon.negative)
        if positive_hits + negative_hits == 0:
            return 0.5

        score = (positive_hits - negative_hits) / (positive_hits + negative_hits)
        return 0.5 + 0.5 * score
