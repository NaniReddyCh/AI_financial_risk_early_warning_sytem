import numpy as np
import pandas as pd

from ai_risk_system.pipeline import FinancialRiskPipeline


def _market_df(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    steps = rng.normal(loc=0.0003, scale=0.01, size=rows)
    prices = 100 * np.cumprod(1 + steps)
    volumes = rng.integers(900000, 1500000, size=rows)
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    return pd.DataFrame({"date": dates, "close": prices, "volume": volumes})


def test_train_and_score():
    pipeline = FinancialRiskPipeline()
    market = _market_df()
    metrics = pipeline.train(market)
    assert "roc_auc" in metrics

    result = pipeline.score_latest(
        market,
        ["Bank warns of possible recession and sector selloff", "Tech earnings beats estimates"],
    )
    assert 0 <= result["final_risk_score"] <= 1
    assert result["risk_category"] in {"Low", "Medium", "High"}
