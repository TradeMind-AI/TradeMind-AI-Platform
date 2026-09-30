from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_validation_predictions.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ONE_WAY_COST = 0.0005

THRESHOLDS = [
    0.00,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


# --------------------------------------------------
# Load validation predictions
# --------------------------------------------------

print("Loading validation predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# --------------------------------------------------
# Sort chronologically
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
)

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Run threshold experiments
# --------------------------------------------------

results = []


for threshold in THRESHOLDS:

    # --------------------------------------------------
    # Determine trading signal
    # --------------------------------------------------

    position = np.zeros(len(df))

    up_signal = (
        (df["probability_up"] >= threshold)
        & (
            df["probability_up"]
            > df["probability_down"]
        )
        & (
            df["probability_up"]
            > df["probability_neutral"]
        )
    )

    down_signal = (
        (df["probability_down"] >= threshold)
        & (
            df["probability_down"]
            > df["probability_up"]
        )
        & (
            df["probability_down"]
            > df["probability_neutral"]
        )
    )

    position[up_signal] = 1
    position[down_signal] = -1

    df["position"] = position

    # --------------------------------------------------
    # Gross return
    # --------------------------------------------------

    df["gross_return"] = (
        df["position"]
        * df["future_return_1h"]
    )

    # --------------------------------------------------
    # Previous position
    # --------------------------------------------------

    previous_position = (
        df["position"]
        .shift(1)
        .fillna(0)
    )

    # --------------------------------------------------
    # Entry / exit / reversal
    # --------------------------------------------------

    entry = (
        (previous_position == 0)
        & (df["position"] != 0)
    )

    exit_position = (
        (previous_position != 0)
        & (df["position"] == 0)
    )

    reversal = (
        (previous_position != 0)
        & (df["position"] != 0)
        & (
            previous_position
            != df["position"]
        )
    )

    # --------------------------------------------------
    # Transaction cost
    # --------------------------------------------------

    transaction_cost = (
        entry.astype(float)
        * ONE_WAY_COST
        +
        exit_position.astype(float)
        * ONE_WAY_COST
        +
        reversal.astype(float)
        * ONE_WAY_COST
        * 2
    )

    df["net_return"] = (
        df["gross_return"]
        - transaction_cost
    )

    # --------------------------------------------------
    # Performance
    # --------------------------------------------------

    gross_total_return = (
        (1 + df["gross_return"]).prod()
        - 1
    )

    net_total_return = (
        (1 + df["net_return"]).prod()
        - 1
    )

    trading_periods = int(
        (df["position"] != 0).sum()
    )

    entries_count = int(
        entry.sum()
    )

    exits_count = int(
        exit_position.sum()
    )

    reversals_count = int(
        reversal.sum()
    )

    total_cost = (
        transaction_cost.sum()
    )

    # --------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------

    equity = (
        1 + df["net_return"]
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity / peak - 1
    )

    max_drawdown = drawdown.min()

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    results.append(
        {
            "threshold": threshold,
            "trading_periods": trading_periods,
            "entries": entries_count,
            "exits": exits_count,
            "reversals": reversals_count,
            "gross_return_pct":
                gross_total_return * 100,
            "transaction_cost_pct":
                total_cost * 100,
            "net_return_pct":
                net_total_return * 100,
            "max_drawdown_pct":
                max_drawdown * 100,
        }
    )


# --------------------------------------------------
# Results table
# --------------------------------------------------

results_df = pd.DataFrame(results)


print("\n" + "=" * 90)
print("VALIDATION CONFIDENCE THRESHOLD RESULTS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# --------------------------------------------------
# Explanation
# --------------------------------------------------

print("\n" + "=" * 90)
print("WHAT THE RESULTS MEAN")
print("=" * 90)

print(
    "\nThreshold 0.00 means we trade whenever "
    "one class has the highest probability."
)

print(
    "\nHigher thresholds mean the model must be "
    "more confident before entering a trade."
)

print(
    "\nWe are using the VALIDATION set only "
    "for this experiment."
)

print("\nExperiment complete.")