from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

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
    / "backtest_results.csv"
)


# --------------------------------------------------
# Backtest configuration
# --------------------------------------------------

# Assumed cost for ONE side of a trade.
#
# Example:
# Entering a position = 0.05%
# Exiting a position = 0.05%
# Complete round trip = 0.10%
#
# This is a research assumption.
# Replace with actual execution costs later.
ONE_WAY_COST = 0.0005

ANNUAL_HOURS = 24 * 365


# --------------------------------------------------
# Load predictions
# --------------------------------------------------

print("Loading prediction data...")

df = pd.read_csv(PREDICTIONS_FILE)

print(f"Rows loaded: {len(df)}")


# --------------------------------------------------
# Validate required columns
# --------------------------------------------------

required_columns = [
    "timestamp",
    "close",
    "prediction",
    "future_return_1h",
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
# Sort chronologically
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# --------------------------------------------------
# Validate timestamps
# --------------------------------------------------

if df["timestamp"].duplicated().any():
    raise ValueError(
        "Duplicate timestamps detected."
    )

if not df["timestamp"].is_monotonic_increasing:
    raise ValueError(
        "Timestamps are not sorted chronologically."
    )


# --------------------------------------------------
# Validate predictions
# --------------------------------------------------

valid_predictions = {
    "UP",
    "DOWN",
    "NEUTRAL",
}

invalid_predictions = set(
    df["prediction"].dropna().unique()
) - valid_predictions

if invalid_predictions:
    raise ValueError(
        f"Unexpected prediction classes: "
        f"{invalid_predictions}"
    )


# --------------------------------------------------
# Convert predictions into positions
# --------------------------------------------------

position_map = {
    "UP": 1,
    "DOWN": -1,
    "NEUTRAL": 0,
}

df["position"] = (
    df["prediction"].map(position_map)
)


# --------------------------------------------------
# Gross strategy return
# --------------------------------------------------

df["gross_strategy_return"] = (
    df["position"]
    * df["future_return_1h"]
)


# --------------------------------------------------
# Calculate position changes
# --------------------------------------------------

df["previous_position"] = (
    df["position"]
    .shift(1)
    .fillna(0)
)


# --------------------------------------------------
# Calculate entry and exit costs
# --------------------------------------------------

# Entering a position:
# previous = 0
# current  = +1 or -1
#
# Exiting a position:
# previous = +1 or -1
# current  = 0
#
# Reversing:
# previous = +1
# current  = -1
# OR
# previous = -1
# current  = +1
#
# A reversal requires BOTH:
# exit old position + enter new position.

df["position_change"] = (
    df["position"]
    - df["previous_position"]
)


df["entry"] = (
    (df["previous_position"] == 0)
    & (df["position"] != 0)
)


df["exit"] = (
    (df["previous_position"] != 0)
    & (df["position"] == 0)
)


df["reversal"] = (
    (df["previous_position"] != 0)
    & (df["position"] != 0)
    & (
        df["previous_position"]
        != df["position"]
    )
)


# --------------------------------------------------
# Transaction costs
# --------------------------------------------------

df["entry_cost"] = np.where(
    df["entry"],
    ONE_WAY_COST,
    0.0,
)

df["exit_cost"] = np.where(
    df["exit"],
    ONE_WAY_COST,
    0.0,
)

df["reversal_cost"] = np.where(
    df["reversal"],
    ONE_WAY_COST * 2,
    0.0,
)


df["transaction_cost"] = (
    df["entry_cost"]
    + df["exit_cost"]
    + df["reversal_cost"]
)


# --------------------------------------------------
# Net strategy return
# --------------------------------------------------

df["net_strategy_return"] = (
    df["gross_strategy_return"]
    - df["transaction_cost"]
)


# --------------------------------------------------
# Buy-and-hold return
# --------------------------------------------------

df["buy_hold_return"] = (
    df["future_return_1h"]
)


# --------------------------------------------------
# Cumulative equity
# --------------------------------------------------

df["strategy_equity"] = (
    1
    * (
        1
        + df["net_strategy_return"]
    ).cumprod()
)


df["buy_hold_equity"] = (
    1
    * (
        1
        + df["buy_hold_return"]
    ).cumprod()
)


# --------------------------------------------------
# Maximum drawdown
# --------------------------------------------------

strategy_peak = (
    df["strategy_equity"]
    .cummax()
)

df["strategy_drawdown"] = (
    df["strategy_equity"]
    / strategy_peak
    - 1
)

max_drawdown = (
    df["strategy_drawdown"].min()
)


# --------------------------------------------------
# Strategy statistics
# --------------------------------------------------

total_return = (
    df["strategy_equity"].iloc[-1]
    - 1
)

buy_hold_total_return = (
    df["buy_hold_equity"].iloc[-1]
    - 1
)


# --------------------------------------------------
# Trade statistics
# --------------------------------------------------

entries = int(
    df["entry"].sum()
)

exits = int(
    df["exit"].sum()
)

reversals = int(
    df["reversal"].sum()
)

total_transaction_events = (
    entries
    + exits
    + (reversals * 2)
)


trading_periods = int(
    (df["position"] != 0).sum()
)


winning_periods = (
    df["net_strategy_return"] > 0
).sum()


if trading_periods > 0:

    win_rate = (
        winning_periods
        / trading_periods
    )

else:

    win_rate = 0.0


# --------------------------------------------------
# Sharpe ratio
# --------------------------------------------------

return_std = (
    df["net_strategy_return"].std()
)

if return_std > 0:

    sharpe_ratio = (
        df["net_strategy_return"].mean()
        / return_std
        * np.sqrt(ANNUAL_HOURS)
    )

else:

    sharpe_ratio = 0.0


# --------------------------------------------------
# Save results
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("CORRECTED BASELINE BACKTEST RESULTS")
print("=" * 60)

print(
    f"\nTest period:"
    f" {df['timestamp'].iloc[0]}"
    f" → {df['timestamp'].iloc[-1]}"
)

print("\nInitial capital: 1.0000")

print(
    f"Final strategy equity: "
    f"{df['strategy_equity'].iloc[-1]:.4f}"
)

print(
    f"Strategy total return: "
    f"{total_return * 100:.2f}%"
)

print(
    f"Buy-and-hold return: "
    f"{buy_hold_total_return * 100:.2f}%"
)

print(
    f"\nTrading periods: "
    f"{trading_periods}"
)

print(
    f"Entries: "
    f"{entries}"
)

print(
    f"Exits: "
    f"{exits}"
)

print(
    f"Reversals: "
    f"{reversals}"
)

print(
    f"Total transaction events: "
    f"{total_transaction_events}"
)

print(
    f"Total transaction cost: "
    f"{df['transaction_cost'].sum() * 100:.2f}%"
)

print(
    f"\nWin rate: "
    f"{win_rate * 100:.2f}%"
)

print(
    f"Maximum drawdown: "
    f"{max_drawdown * 100:.2f}%"
)

print(
    f"Sharpe ratio: "
    f"{sharpe_ratio:.4f}"
)

print(
    f"\nOne-way transaction cost assumption: "
    f"{ONE_WAY_COST * 100:.2f}%"
)

print(
    f"Round-trip transaction cost assumption: "
    f"{ONE_WAY_COST * 2 * 100:.2f}%"
)

print(
    f"\nDetailed results saved to:"
    f"\n{OUTPUT_FILE}"
)

print("\nCorrected backtest complete.")