from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PAPER_LOG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_trades.csv"
)


def main():
    if not PAPER_LOG.exists():
        raise FileNotFoundError(
            f"Paper trade log not found:\n{PAPER_LOG}"
        )

    df = pd.read_csv(PAPER_LOG)

    if df.empty:
        print("Paper trade log is empty.")
        return

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    # Make sure optional settlement columns exist.
    optional_columns = [
        "status",
        "result",
        "net_return_pct",
        "gross_return_pct",
        "entry_price",
        "exit_price",
        "cumulative_net_return_pct",
    ]

    for column in optional_columns:
        if column not in df.columns:
            df[column] = pd.NA

    total_signals = len(df)

    buy_count = (
        df["signal"] == "BUY"
    ).sum()

    sell_count = (
        df["signal"] == "SELL"
    ).sum()

    hold_count = (
        df["signal"] == "HOLD"
    ).sum()

    closed = df[
        df["status"] == "CLOSED"
    ].copy()

    open_trades = df[
        (
            df["status"] == "OPEN"
        )
        & df["signal"].isin(
            [
                "BUY",
                "SELL",
            ]
        )
    ].copy()

    wins = (
        closed["result"] == "WIN"
    ).sum()

    losses = (
        closed["result"] == "LOSS"
    ).sum()

    total_closed = len(closed)

    win_rate = (
        wins
        / total_closed
        * 100
        if total_closed > 0
        else 0.0
    )

    closed["net_return_pct"] = pd.to_numeric(
        closed["net_return_pct"],
        errors="coerce",
    )

    winning_returns = closed[
        closed["net_return_pct"] > 0
    ]["net_return_pct"]

    losing_returns = closed[
        closed["net_return_pct"] <= 0
    ]["net_return_pct"]

    average_win = (
        winning_returns.mean()
        if not winning_returns.empty
        else 0.0
    )

    average_loss = (
        losing_returns.mean()
        if not losing_returns.empty
        else 0.0
    )

    best_trade = (
        closed["net_return_pct"].max()
        if not closed.empty
        else 0.0
    )

    worst_trade = (
        closed["net_return_pct"].min()
        if not closed.empty
        else 0.0
    )

    cumulative_return = (
        closed["net_return_pct"]
        .fillna(0)
        .sum()
        if not closed.empty
        else 0.0
    )

    latest = (
        df.sort_values("timestamp")
        .iloc[-1]
    )

    print("\n" + "=" * 72)
    print("TRADEMIND AI - PAPER TRADING REPORT")
    print("=" * 72)

    print(
        f"Total signals:            {total_signals}"
    )

    print(
        f"BUY signals:              {buy_count}"
    )

    print(
        f"SELL signals:             {sell_count}"
    )

    print(
        f"HOLD signals:             {hold_count}"
    )

    print()

    print(
        f"Closed trades:            {total_closed}"
    )

    print(
        f"Open trades:              {len(open_trades)}"
    )

    print(
        f"Wins:                     {wins}"
    )

    print(
        f"Losses:                   {losses}"
    )

    print(
        f"Win rate:                 {win_rate:.2f}%"
    )

    print()

    print(
        f"Average winning trade:    {average_win:.2f}%"
    )

    print(
        f"Average losing trade:     {average_loss:.2f}%"
    )

    print(
        f"Best trade:               {best_trade:.2f}%"
    )

    print(
        f"Worst trade:              {worst_trade:.2f}%"
    )

    print(
        f"Cumulative net return:    {cumulative_return:.2f}%"
    )

    print("\n" + "-" * 72)
    print("LATEST SIGNAL")
    print("-" * 72)

    print(
        f"Timestamp:                 {latest['timestamp']}"
    )

    print(
        f"Price:                     {latest['price']}"
    )

    print(
        f"Prediction:                {latest['prediction']}"
    )

    print(
        f"Confidence:                {float(latest['confidence']):.4f}"
    )

    print(
        f"Signal:                    {latest['signal']}"
    )

    print(
        f"Status:                    {latest['status']}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()