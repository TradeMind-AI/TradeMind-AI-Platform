from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_features.csv"
)

DERIVATIVES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_derivatives_1h.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_features_enriched.csv"
)


def main():
    print("Loading BTC feature dataset...")
    features = pd.read_csv(FEATURE_FILE)

    print("Loading derivatives dataset...")
    derivatives = pd.read_csv(DERIVATIVES_FILE)

    features["open_time"] = pd.to_datetime(
        features["open_time"],
        utc=True,
    )

    derivatives["timestamp"] = pd.to_datetime(
        derivatives["timestamp"],
        utc=True,
    )

    derivatives = derivatives.rename(
        columns={
            "timestamp": "open_time"
        }
    )

    print(f"Feature rows:     {len(features)}")
    print(f"Derivative rows:  {len(derivatives)}")

    merged = features.merge(
        derivatives[
            [
                "open_time",
                "funding_rate",
            ]
        ],
        on="open_time",
        how="left",
        validate="one_to_one",
    )

    missing_before = (
        merged["funding_rate"]
        .isna()
        .sum()
    )

    print(
        f"Missing funding rows after merge: "
        f"{missing_before}"
    )

    # Only carry information forward.
    # Never backfill because that could introduce future data.
    merged["funding_rate"] = (
        merged["funding_rate"]
        .ffill()
    )

    missing_after = (
        merged["funding_rate"]
        .isna()
        .sum()
    )

    print(
        f"Missing funding rows after forward fill: "
        f"{missing_after}"
    )

    if missing_after > 0:
        merged = merged.dropna(
            subset=["funding_rate"]
        ).copy()

    # Additional causal funding features
    merged["funding_rate_change_1h"] = (
        merged["funding_rate"]
        .diff()
    )

    merged["funding_rate_change_8h"] = (
        merged["funding_rate"]
        - merged["funding_rate"].shift(8)
    )

    merged["funding_rate_mean_24h"] = (
        merged["funding_rate"]
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
    )

    merged["funding_rate_abs"] = (
        merged["funding_rate"].abs()
    )

    merged = merged.dropna().copy()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nMerge complete.")
    print(f"Final rows: {len(merged)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nFunding features added:")
    print(
        [
            "funding_rate",
            "funding_rate_change_1h",
            "funding_rate_change_8h",
            "funding_rate_mean_24h",
            "funding_rate_abs",
        ]
    )


if __name__ == "__main__":
    main()