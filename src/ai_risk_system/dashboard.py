from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from .data_sources import fetch_market_data
from .pipeline import FinancialRiskPipeline


@dataclass
class DashboardState:
    symbol: str
    model_path: Path
    period: str


def _category_color(category: str) -> str:
    mapping = {
        "Low": "#2e7d32",
        "Medium": "#f9a825",
        "High": "#c62828",
    }
    return mapping.get(category, "#546e7a")


def _render_risk_card(output: dict) -> None:
    category = output["risk_category"]
    color = _category_color(category)
    st.markdown(
        f"""
<div style='padding:16px;border-radius:12px;background:#111827;color:white;border-left:8px solid {color};'>
  <h3 style='margin:0;'>Current Risk: {category}</h3>
  <p style='margin:8px 0 0 0;font-size:18px;'>Final Score: <b>{output['final_risk_score']:.3f}</b></p>
  <p style='margin:4px 0 0 0;'>Market: {output['market_risk_score']:.3f} | Sentiment: {output['sentiment_score']:.3f}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _make_trend_frame(market_df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    trend = market_df[["date", close_col]].copy()
    trend["rolling_20"] = trend[close_col].rolling(20).mean()
    return trend.dropna()


def run_dashboard() -> None:
    st.set_page_config(page_title="AI Financial Risk EWS", page_icon="⚠️", layout="wide")
    st.title("⚠️ AI Financial Risk Early Warning Dashboard")
    st.caption("Uses your pretrained market model + pretrained sentiment inference (no retraining on score run).")

    with st.sidebar:
        st.header("Configuration")
        symbol = st.text_input("Yahoo Symbol", value="^GSPC")
        period = st.selectbox("Market History Window", options=["6mo", "1y", "2y", "5y"], index=1)
        model_path = Path(st.text_input("Pretrained Market Model Path", value="artifacts/risk_model.joblib"))
        headlines_text = st.text_area(
            "News Headlines (one per line)",
            value="Markets mixed as investors await inflation data\nMajor bank warns of recession risks",
            height=160,
        )
        score_clicked = st.button("Run Daily Risk Score", type="primary", use_container_width=True)

    state = DashboardState(symbol=symbol, model_path=model_path, period=period)
    headlines = [line.strip() for line in headlines_text.splitlines() if line.strip()]

    st.info(
        "Market model features expected by default: ret_1d, ret_5d, volatility_14, ma20, ma50, price_ma_ratio. "
        "You can train externally and place the model at the configured path."
    )

    if score_clicked:
        if not state.model_path.exists():
            st.error("Model file not found. Please provide your externally trained market model path.")
            return

        with st.spinner("Loading model and scoring latest data..."):
            pipeline = FinancialRiskPipeline.load_model(state.model_path)
            market_df = fetch_market_data(state.symbol, period=state.period)
            output = pipeline.score_latest(market_df, headlines)

        _render_risk_card(output)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Risk Components")
            st.bar_chart(
                pd.DataFrame(
                    {
                        "score": [output["market_risk_score"], output["sentiment_score"], output["final_risk_score"]]
                    },
                    index=["Market", "Sentiment", "Final"],
                )
            )
        with col2:
            st.subheader("Recent Market Trend")
            price_col = "adj_close" if "adj_close" in market_df.columns else "close"
            trend_df = _make_trend_frame(market_df, close_col=price_col)
            st.line_chart(trend_df.set_index("date")[[price_col, "rolling_20"]])

        st.subheader("Headlines Used")
        st.write("\n".join([f"- {h}" for h in headlines]) if headlines else "No headlines provided.")


if __name__ == "__main__":
    run_dashboard()
