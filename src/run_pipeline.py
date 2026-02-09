import argparse

from src.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the risk early-warning model.")
    parser.add_argument("--market", required=True, help="Path to market CSV.")
    parser.add_argument("--news", required=True, help="Path to news CSV.")
    parser.add_argument("--output", default="reports", help="Output directory for artifacts.")
    parser.add_argument("--horizon", type=int, default=10, help="Risk horizon in days.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=-0.05,
        help="Future return threshold to label high risk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        risk_horizon_days=args.horizon,
        risk_drop_threshold=args.threshold,
    )
    run_pipeline(args.market, args.news, args.output, config)


if __name__ == "__main__":
    main()
