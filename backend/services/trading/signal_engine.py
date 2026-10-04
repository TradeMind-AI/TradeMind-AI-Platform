from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trading_signals.csv"
)


CONFIDENCE_THRESHOLD = 0.70


def generate_signal(row: pd.Series) -> str:
    prediction = row["prediction"]

    probability_up = row["probability_up"]
    probability_down = row["probability_down"]
    probability_neutral = row["probability_neutral"]

    max_probability = max(
        probability_up,
        probability_down,
        probability_neutral,
    )

    if max_probability < CONFIDENCE_THRESHOLD:
        return "HOLD"

    if prediction == "UP":
        return "BUY"

    if prediction == "DOWN":
        return "SELL"

    return "HOLD"


def main():
    print("Loading ML predictions...")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [
        "timestamp",
        "close",
        "prediction",
        "probability_down",
        "probability_neutral",
        "probability_up",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print(f"Rows loaded: {len(df)}")

    df["signal"] = df.apply(
        generate_signal,
        axis=1,
    )

    df["confidence"] = df[
        [
            "probability_down",
            "probability_neutral",
            "probability_up",
        ]
    ].max(axis=1)

    output_columns = [
        "timestamp",
        "close",
        "prediction",
        "confidence",
        "signal",
        "probability_down",
        "probability_neutral",
        "probability_up",
    ]

    signal_df = df[output_columns].copy()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    signal_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nTrading signal generation complete.")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nSignal distribution:")
    print(
        signal_df["signal"]
        .value_counts()
    )

    print("\nFirst 10 signals:")
    print(
        signal_df.head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()