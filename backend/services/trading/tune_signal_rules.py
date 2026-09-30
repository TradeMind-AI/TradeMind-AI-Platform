from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_validation_predictions.csv"
)

TRANSACTION_COST_PCT = 0.05

CONFIDENCE_THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

MARGIN_THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]


def generate_signal(
    row: pd.Series,
    confidence_threshold: float,
    margin_threshold: float,
) -> str:
    prediction = row["prediction"]

    up = row["probability_up"]
    down = row["probability_down"]

    if prediction == "UP":
        confidence = up
        margin = up - down

        if (
            confidence >= confidence_threshold
            and margin >= margin_threshold
        ):
            return "BUY"

    if prediction == "DOWN":
        confidence = down
        margin = down - up

        if (
            confidence >= confidence_threshold
            and margin >= margin_threshold
        ):
            return "SELL"

    return "HOLD"


def evaluate_strategy(
    df: pd.DataFrame,
    confidence_threshold: float,
    margin_threshold: float,
) -> dict:
    working_df = df.copy()

    working_df["signal"] = working_df.apply(
        lambda row: generate_signal(
            row,
            confidence_threshold,
            margin_threshold,
        ),
        axis=1,
    )

    working_df["next_close"] = (
        working_df["close"].shift(-1)
    )

    working_df["future_return"] = (
        working_df["next_close"]
        / working_df["close"]
        - 1
    )

    trade_mask = working_df["signal"].isin(
        ["BUY", "SELL"]
    )

    trades = working_df[trade_mask].copy()

    if trades.empty:
        return {
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
            "trades": 0,
            "win_rate_pct": 0.0,
            "gross_return_pct": 0.0,
            "transaction_cost_pct": 0.0,
            "net_return_pct": 0.0,
        }

    trades["trade_return"] = 0.0

    buy_mask = trades["signal"] == "BUY"
    sell_mask = trades["signal"] == "SELL"

    trades.loc[
        buy_mask,
        "trade_return"
    ] = trades.loc[
        buy_mask,
        "future_return"
    ]

    trades.loc[
        sell_mask,
        "trade_return"
    ] = -trades.loc[
        sell_mask,
        "future_return"
    ]

    trades = trades.dropna(
        subset=["trade_return"]
    )

    gross_return_pct = (
        trades["trade_return"].sum()
        * 100
    )

    transaction_cost_pct = (
        len(trades)
        * TRANSACTION_COST_PCT
    )

    net_return_pct = (
        gross_return_pct
        - transaction_cost_pct
    )

    winning_trades = (
        trades["trade_return"]
        - TRANSACTION_COST_PCT / 100
        > 0
    ).sum()

    win_rate_pct = (
        winning_trades
        / len(trades)
        * 100
        if len(trades) > 0
        else 0
    )

    return {
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "trades": len(trades),
        "win_rate_pct": win_rate_pct,
        "gross_return_pct": gross_return_pct,
        "transaction_cost_pct": transaction_cost_pct,
        "net_return_pct": net_return_pct,
    }


def main():
    print("Loading validation predictions...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Validation prediction file not found: "
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df)}")

    results = []

    for confidence_threshold in CONFIDENCE_THRESHOLDS:
        for margin_threshold in MARGIN_THRESHOLDS:
            result = evaluate_strategy(
                df,
                confidence_threshold,
                margin_threshold,
            )

            results.append(result)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "net_return_pct",
            "win_rate_pct",
        ],
        ascending=[
            False,
            False,
        ],
    )

    print("\n" + "=" * 110)
    print("SIGNAL RULE TUNING — VALIDATION SET")
    print("=" * 110)

    display_df = results_df.copy()

    for column in [
        "win_rate_pct",
        "gross_return_pct",
        "transaction_cost_pct",
        "net_return_pct",
    ]:
        display_df[column] = (
            display_df[column]
            .map(lambda value: f"{value:.2f}%")
        )

    print(
        display_df.head(25)
        .to_string(index=False)
    )

    print("\n" + "=" * 110)

    best = results_df.iloc[0]

    print("BEST VALIDATION RULE")
    print(
        f"Confidence threshold: "
        f"{best['confidence_threshold']:.2f}"
    )
    print(
        f"Margin threshold:     "
        f"{best['margin_threshold']:.2f}"
    )
    print(
        f"Trades:               "
        f"{int(best['trades'])}"
    )
    print(
        f"Win rate:             "
        f"{best['win_rate_pct']:.2f}%"
    )
    print(
        f"Net return:           "
        f"{best['net_return_pct']:.2f}%"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()