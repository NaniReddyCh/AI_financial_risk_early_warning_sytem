from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_sources import fetch_market_data
from .pipeline import FinancialRiskPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Financial Risk Early Warning System")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Optional: train market model locally")
    train_parser.add_argument("--symbol", default="^GSPC", help="Yahoo symbol (default: S&P 500)")
    train_parser.add_argument("--period", default="5y", help="History period")
    train_parser.add_argument("--model-path", default="artifacts/risk_model.joblib")

    score_parser = sub.add_parser("score", help="Run daily risk score using pretrained market model")
    score_parser.add_argument("--symbol", default="^GSPC")
    score_parser.add_argument("--period", default="1y")
    score_parser.add_argument("--model-path", default="artifacts/risk_model.joblib")
    score_parser.add_argument(
        "--headline",
        action="append",
        default=[],
        help="Provide one or more news headlines (repeat flag)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "train":
        market_df = fetch_market_data(args.symbol, period=args.period)
        pipeline = FinancialRiskPipeline()
        metrics = pipeline.train(market_df)
        pipeline.save_model(Path(args.model_path))
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "score":
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pretrained market model not found at {model_path}. "
                "Provide your externally trained model path via --model-path."
            )

        pipeline = FinancialRiskPipeline.load_model(model_path)
        market_df = fetch_market_data(args.symbol, period=args.period)
        if not args.headline:
            args.headline = [
                "Markets mixed as investors await inflation data",
                "Major bank warns of recession risks in next quarter",
            ]
        output = pipeline.score_latest(market_df, args.headline)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
