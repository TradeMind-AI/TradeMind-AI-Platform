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
    / "backtest_results.csv"
)


# --------------------------------------------------
# Load backtest results
# --------------------------------------------------

print("Loading backtest results...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# --------------------------------------------------
# Basic validation
# --------------------------------------------------

required_columns = [
    "timestamp",
    "prediction",
    "position",
    "future_return_1h",
    "gross_strategy_return",
    "transaction_cost",
    "net_strategy_return",
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


# --------------------------------------------------
# Overall return analysis
# --------------------------------------------------

gross_total_return = (
    (1 + df["gross_strategy_return"]).prod() - 1
)

net_total_return = (
    (1 + df["net_strategy_return"]).prod() - 1
)

total_transaction_cost = (
    df["transaction_cost"].sum()
)


print("\n" + "=" * 60)
print("OVERALL RETURN ANALYSIS")
print("=" * 60)

print(
    f"\nGross strategy return: "
    f"{gross_total_return * 100:.2f}%"
)

print(
    f"Net strategy return: "
    f"{net_total_return * 100:.2f}%"
)

print(
    f"Total transaction costs: "
    f"{total_transaction_cost * 100:.2f}%"
)


# --------------------------------------------------
# Prediction distribution
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

prediction_counts = (
    df["prediction"]
    .value_counts()
    .reindex(["UP", "DOWN", "NEUTRAL"])
    .fillna(0)
    .astype(int)
)

print(prediction_counts)

prediction_percentages = (
    prediction_counts / len(df) * 100
)

print("\nPrediction percentages:")

for prediction, percentage in prediction_percentages.items():
    print(
        f"{prediction}: "
        f"{percentage:.2f}%"
    )


# --------------------------------------------------
# Performance by prediction
# --------------------------------------------------

print("\n" + "=" * 60)
print("PERFORMANCE BY PREDICTION")
print("=" * 60)

for prediction in ["UP", "DOWN", "NEUTRAL"]:

    subset = df[
        df["prediction"] == prediction
    ]

    gross_return = (
        (1 + subset["gross_strategy_return"]).prod() - 1
    )

    net_return = (
        (1 + subset["net_strategy_return"]).prod() - 1
    )

    average_future_return = (
        subset["future_return_1h"].mean()
    )

    profitable_periods = (
        subset["net_strategy_return"] > 0
    ).sum()

    total_periods = len(subset)

    if total_periods > 0:
        win_rate = (
            profitable_periods
            / total_periods
        )
    else:
        win_rate = 0.0

    print(f"\n{prediction}")
    print("-" * 30)

    print(
        f"Periods: {total_periods}"
    )

    print(
        f"Average actual BTC return: "
        f"{average_future_return * 100:.4f}%"
    )

    print(
        f"Gross return contribution: "
        f"{gross_return * 100:.2f}%"
    )

    print(
        f"Net return contribution: "
        f"{net_return * 100:.2f}%"
    )

    print(
        f"Win rate: "
        f"{win_rate * 100:.2f}%"
    )


# --------------------------------------------------
# Long vs short analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("LONG VS SHORT ANALYSIS")
print("=" * 60)

for position_name, position_value in [
    ("LONG", 1),
    ("SHORT", -1),
    ("NO TRADE", 0),
]:

    subset = df[
        df["position"] == position_value
    ]

    if len(subset) == 0:
        continue

    gross_return = (
        (1 + subset["gross_strategy_return"]).prod() - 1
    )

    average_strategy_return = (
        subset["gross_strategy_return"].mean()
    )

    profitable_periods = (
        subset["gross_strategy_return"] > 0
    ).sum()

    win_rate = (
        profitable_periods
        / len(subset)
    )

    print(f"\n{position_name}")
    print("-" * 30)

    print(
        f"Periods: {len(subset)}"
    )

    print(
        f"Gross return: "
        f"{gross_return * 100:.2f}%"
    )

    print(
        f"Average return per period: "
        f"{average_strategy_return * 100:.4f}%"
    )

    print(
        f"Win rate: "
        f"{win_rate * 100:.2f}%"
    )


# --------------------------------------------------
# Transaction cost analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRANSACTION COST ANALYSIS")
print("=" * 60)

trade_count = int(
    df["trade"].sum()
)

print(
    f"\nNumber of position changes: "
    f"{trade_count}"
)

print(
    f"Cost per position change: "
    f"0.10%"
)

print(
    f"Total transaction cost: "
    f"{total_transaction_cost * 100:.2f}%"
)


# --------------------------------------------------
# Largest losing periods
# --------------------------------------------------

print("\n" + "=" * 60)
print("LARGEST LOSING PERIODS")
print("=" * 60)

largest_losses = (
    df[
        [
            "timestamp",
            "prediction",
            "future_return_1h",
            "gross_strategy_return",
            "transaction_cost",
            "net_strategy_return",
        ]
    ]
    .sort_values(
        "net_strategy_return"
    )
    .head(10)
)

print(
    largest_losses.to_string(
        index=False
    )
)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)

print(
    f"\nGross return: "
    f"{gross_total_return * 100:.2f}%"
)

print(
    f"Net return: "
    f"{net_total_return * 100:.2f}%"
)

print(
    f"Transaction cost impact: "
    f"{total_transaction_cost * 100:.2f}%"
)

print(
    f"Total position changes: "
    f"{trade_count}"
)

print("\nAnalysis complete.")