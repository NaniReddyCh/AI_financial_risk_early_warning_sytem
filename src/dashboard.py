import argparse
from pathlib import Path

import pandas as pd
import streamlit as st


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the risk dashboard.")
    parser.add_argument("--predictions", default="reports/risk_predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        st.error("Predictions file not found. Run the pipeline first.")
        return

    data = pd.read_csv(predictions_path)
    st.title("AI Financial Risk Early Warning Dashboard")
    st.metric("Latest Risk Probability", f"{data['risk_probability'].iloc[-1]:.2%}")

    st.subheader("Risk Probability Over Time")
    st.line_chart(data.set_index("date")["risk_probability"])

    st.subheader("Recent Alerts")
    alerts = data.sort_values("date", ascending=False).head(10)
    st.dataframe(alerts)


if __name__ == "__main__":
    main()
