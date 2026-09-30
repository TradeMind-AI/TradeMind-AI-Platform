from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_validation_predictions.csv"
)


# ---------------------------------------------------------
# Trading settings
# ---------------------------------------------------------

ONE_WAY_COST = 0.0005  # 0.05% per entry/exit


# ---------------------------------------------------------
# Load validation predictions
# ---------------------------------------------------------

print("Loading validation predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# ---------------------------------------------------------
# Convert prediction into trading position
# ---------------------------------------------------------

# UP      -> LONG  (+1)
# DOWN    -> SHORT (-1)
# NEUTRAL -> NO POSITION (0)

df["position"] = 0

df.loc[df["prediction"] == "UP", "position"] = 1
df.loc[df["prediction"] == "DOWN", "position"] = -1


# ---------------------------------------------------------
# Calculate gross strategy return
# ---------------------------------------------------------

df["gross_return"] = (
    df["position"] * df["future_return_1h"]
)


# ---------------------------------------------------------
# Calculate transaction costs
# ---------------------------------------------------------

previous_position = df["position"].shift(1).fillna(0)

df["position_change"] = (
    df["position"] - previous_position
)


# Number of one-way transactions:
# 0 -> LONG  = 1
# 0 -> SHORT = 1
# LONG -> 0  = 1
# SHORT -> 0 = 1
# LONG -> SHORT = 2
# SHORT -> LONG = 2

df["transaction_events"] = df["position_change"].abs()

df["transaction_cost"] = (
    df["transaction_events"] * ONE_WAY_COST
)


# ---------------------------------------------------------
# Net return
# ---------------------------------------------------------

df["net_return"] = (
    df["gross_return"] - df["transaction_cost"]
)


# ---------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------

trading_periods = (df["position"] != 0).sum()

entries = (
    ((previous_position == 0) & (df["position"] != 0))
).sum()

exits = (
    ((previous_position != 0) & (df["position"] == 0))
).sum()

reversals = (
    (
        (previous_position != 0)
        & (df["position"] != 0)
        & (previous_position != df["position"])
    )
).sum()

total_transactions = df["transaction_events"].sum()

gross_return = df["gross_return"].sum()

total_cost = df["transaction_cost"].sum()

net_return = df["net_return"].sum()


# ---------------------------------------------------------
# Win rate
# ---------------------------------------------------------

active_trades = df[df["position"] != 0]

winning_periods = (
    active_trades["gross_return"] > 0
).sum()

win_rate = (
    winning_periods / len(active_trades)
    if len(active_trades) > 0
    else 0
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print()
print("=" * 80)
print("VALIDATION TRADING ANALYSIS")
print("=" * 80)

print(f"Trading periods:       {trading_periods}")
print(f"Entries:               {entries}")
print(f"Exits:                 {exits}")
print(f"Reversals:             {reversals}")
print(f"Transaction events:    {total_transactions}")

print()
print(f"Gross return:          {gross_return * 100:.2f}%")
print(f"Transaction cost:      {total_cost * 100:.2f}%")
print(f"Net return:            {net_return * 100:.2f}%")
print(f"Win rate:              {win_rate * 100:.2f}%")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)