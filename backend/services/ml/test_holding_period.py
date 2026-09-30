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

HOLDING_PERIODS = [3, 6, 12, 24]


print("Loading validation predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# ---------------------------------------------------------
# Convert prediction to direction
# UP = LONG
# DOWN = SHORT
# NEUTRAL = NO TRADE
# ---------------------------------------------------------

df["raw_position"] = 0

df.loc[df["prediction"] == "UP", "raw_position"] = 1
df.loc[df["prediction"] == "DOWN", "raw_position"] = -1


results = []


# ---------------------------------------------------------
# Test each minimum holding period
# ---------------------------------------------------------

for holding_period in HOLDING_PERIODS:

    position = 0
    candles_held = 0

    positions = []

    for i in range(len(df)):

        signal = df.iloc[i]["raw_position"]

        # No current position
        if position == 0:

            if signal != 0:
                position = signal
                candles_held = 1

        # Currently holding a position
        else:

            candles_held += 1

            # We can only change position after the
            # minimum holding period has passed.
            if candles_held > holding_period:

                # Opposite signal -> reverse
                if signal != 0 and signal != position:
                    position = signal
                    candles_held = 1

                # Neutral -> exit
                elif signal == 0:
                    position = 0
                    candles_held = 0

        positions.append(position)

    df["position"] = positions

    # -----------------------------------------------------
    # Returns
    # -----------------------------------------------------

    df["gross_return"] = (
        df["position"]
        * df["future_return_1h"]
    )

    previous_position = (
        df["position"]
        .shift(1)
        .fillna(0)
    )

    df["transaction_events"] = (
        df["position"]
        - previous_position
    ).abs()

    df["transaction_cost"] = (
        df["transaction_events"]
        * ONE_WAY_COST
    )

    df["net_return"] = (
        df["gross_return"]
        - df["transaction_cost"]
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

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
        & (
            previous_position
            != df["position"]
        )
    ).sum()

    gross_return = df["gross_return"].sum()

    transaction_cost = (
        df["transaction_cost"].sum()
    )

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

    results.append({
        "holding_period_hours": holding_period,
        "trading_periods": trading_periods,
        "entries": entries,
        "exits": exits,
        "reversals": reversals,
        "gross_return_pct": gross_return * 100,
        "transaction_cost_pct": transaction_cost * 100,
        "net_return_pct": net_return * 100,
        "win_rate_pct": win_rate * 100,
    })


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print()
print("=" * 110)
print("MINIMUM HOLDING PERIOD COMPARISON")
print("=" * 110)

print(
    results_df.to_string(
        index=False,
        formatters={
            "gross_return_pct": "{:.2f}%".format,
            "transaction_cost_pct": "{:.2f}%".format,
            "net_return_pct": "{:.2f}%".format,
            "win_rate_pct": "{:.2f}%".format,
        },
    )
)

print()
print("=" * 110)
print("ANALYSIS COMPLETE")
print("=" * 110)