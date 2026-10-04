from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SIGNALS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trading_signals.csv"
)

TRANSACTION_COST_PCT = 0.05


def main():
    print("Loading trading signals...")

    if not SIGNALS_FILE.exists():
        raise FileNotFoundError(
            f"Trading signals file not found: {SIGNALS_FILE}"
        )

    df = pd.read_csv(SIGNALS_FILE)

    required_columns = [
        "timestamp",
        "close",
        "signal",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["next_close"] = df["close"].shift(-1)

    df["future_return"] = (
        df["next_close"] / df["close"] - 1
    )

    df["strategy_return"] = 0.0

    df.loc[
        df["signal"] == "BUY",
        "strategy_return"
    ] = df["future_return"]

    df.loc[
        df["signal"] == "SELL",
        "strategy_return"
    ] = -df["future_return"]

    trade_mask = df["signal"].isin(
        ["BUY", "SELL"]
    )

    df.loc[
        trade_mask,
        "strategy_return"
    ] -= TRANSACTION_COST_PCT / 100

    trade_df = df[trade_mask].copy()

    total_trades = len(trade_df)

    winning_trades = (
        trade_df["strategy_return"] > 0
    ).sum()

    win_rate = (
        winning_trades / total_trades * 100
        if total_trades > 0
        else 0
    )

    gross_return = (
        trade_df["future_return"]
        .where(
            trade_df["signal"] == "BUY",
            -trade_df["future_return"],
        )
        .sum()
        * 100
    )

    transaction_cost = (
        total_trades
        * TRANSACTION_COST_PCT
    )

    net_return = (
        trade_df["strategy_return"].sum()
        * 100
    )

    print("\n" + "=" * 70)
    print("TRADING SIGNAL EVALUATION")
    print("=" * 70)

    print(f"Total rows:        {len(df)}")
    print(f"BUY signals:       {(df['signal'] == 'BUY').sum()}")
    print(f"SELL signals:      {(df['signal'] == 'SELL').sum()}")
    print(f"HOLD signals:      {(df['signal'] == 'HOLD').sum()}")

    print(f"\nTotal trades:      {total_trades}")
    print(f"Winning trades:    {winning_trades}")
    print(f"Win rate:          {win_rate:.2f}%")

    print(f"\nGross return:      {gross_return:.2f}%")
    print(f"Transaction cost:  {transaction_cost:.2f}%")
    print(f"Net return:        {net_return:.2f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()