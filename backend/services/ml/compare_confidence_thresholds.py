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

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


print("Loading validation predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


results = []


for threshold in THRESHOLDS:

    # Start with no position
    df["position"] = 0

    # Trade only when the required probability is high enough
    df.loc[
        (df["probability_up"] >= threshold)
        & (df["probability_up"] == df[
            ["probability_down", "probability_neutral", "probability_up"]
        ].max(axis=1)),
        "position"
    ] = 1

    df.loc[
        (df["probability_down"] >= threshold)
        & (df["probability_down"] == df[
            ["probability_down", "probability_neutral", "probability_up"]
        ].max(axis=1)),
        "position"
    ] = -1

    # Strategy return
    df["gross_return"] = (
        df["position"] * df["future_return_1h"]
    )

    # Previous position
    previous_position = (
        df["position"].shift(1).fillna(0)
    )

    # Transaction events
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

    # Statistics
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

    results.append({
        "threshold": threshold,
        "trading_periods": trading_periods,
        "entries": entries,
        "exits": exits,
        "reversals": reversals,
        "gross_return_pct": gross_return * 100,
        "transaction_cost_pct": transaction_cost * 100,
        "net_return_pct": net_return * 100,
        "win_rate_pct": win_rate * 100,
    })


results_df = pd.DataFrame(results)


print()
print("=" * 100)
print("CONFIDENCE THRESHOLD COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "gross_return_pct": "{:.2f}%".format,
            "transaction_cost_pct": "{:.2f}%".format,
            "net_return_pct": "{:.2f}%".format,
            "win_rate_pct": "{:.2f}%".format,
        },
    )
)

print()
print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)