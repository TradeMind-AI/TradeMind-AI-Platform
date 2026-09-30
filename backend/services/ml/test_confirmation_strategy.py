from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_validation_predictions.csv"
)

ONE_WAY_COST = 0.0005


print("Loading validation predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# Convert predictions to numbers
# UP = 1
# DOWN = -1
# NEUTRAL = 0

df["raw_position"] = 0

df.loc[df["prediction"] == "UP", "raw_position"] = 1
df.loc[df["prediction"] == "DOWN", "raw_position"] = -1


# ---------------------------------------------------------
# 2-candle confirmation
# ---------------------------------------------------------

df["position"] = 0

for i in range(1, len(df)):

    current = df.iloc[i]["raw_position"]
    previous = df.iloc[i - 1]["raw_position"]

    # Two consecutive UP predictions
    if current == 1 and previous == 1:
        df.loc[i, "position"] = 1

    # Two consecutive DOWN predictions
    elif current == -1 and previous == -1:
        df.loc[i, "position"] = -1

    # Otherwise:
    # position remains 0 = NO TRADE


# ---------------------------------------------------------
# Returns
# ---------------------------------------------------------

df["gross_return"] = (
    df["position"] * df["future_return_1h"]
)


previous_position = (
    df["position"].shift(1).fillna(0)
)

df["transaction_events"] = (
    df["position"] - previous_position
).abs()

df["transaction_cost"] = (
    df["transaction_events"] * ONE_WAY_COST
)

df["net_return"] = (
    df["gross_return"]
    - df["transaction_cost"]
)


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

trading_periods = (
    df["position"] != 0
).sum()

entries = (
    (previous_position == 0)
    & (df["position"] != 0)
).sum()

exits = (
    (previous_position != 0)
    & (df["position"] == 0)
).sum()

reversals = (
    (previous_position != 0)
    & (df["position"] != 0)
    & (previous_position != df["position"])
).sum()

transaction_events = (
    df["transaction_events"].sum()
)

gross_return = df["gross_return"].sum()

transaction_cost = df["transaction_cost"].sum()

net_return = df["net_return"].sum()


active_periods = df[
    df["position"] != 0
]

if len(active_periods) > 0:
    win_rate = (
        active_periods["gross_return"] > 0
    ).mean()
else:
    win_rate = 0


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 80)
print("2-CANDLE CONFIRMATION STRATEGY")
print("=" * 80)

print(f"Trading periods:       {trading_periods}")
print(f"Entries:               {entries}")
print(f"Exits:                 {exits}")
print(f"Reversals:             {reversals}")
print(f"Transaction events:    {transaction_events}")

print()
print(f"Gross return:          {gross_return * 100:.2f}%")
print(f"Transaction cost:      {transaction_cost * 100:.2f}%")
print(f"Net return:            {net_return * 100:.2f}%")
print(f"Win rate:              {win_rate * 100:.2f}%")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)