from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PAPER_LOG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_trades.csv"
)

BTC_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "btcusdt_1h.csv"
)

TRANSACTION_COST_PCT = 0.05


def load_paper_log():
    if not PAPER_LOG.exists():
        raise FileNotFoundError(
            f"Paper log not found:\n{PAPER_LOG}"
        )

    df = pd.read_csv(PAPER_LOG)

    if df.empty:
        raise ValueError(
            "Paper trade log is empty."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    return df


def load_btc():
    if not BTC_FILE.exists():
        raise FileNotFoundError(
            f"BTC file not found:\n{BTC_FILE}"
        )

    btc = pd.read_csv(BTC_FILE)

    btc["open_time"] = pd.to_datetime(
        btc["open_time"],
        utc=True,
    )

    btc["close"] = pd.to_numeric(
        btc["close"],
        errors="coerce",
    )

    btc = (
        btc.sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return btc


def main():
    print("Loading paper trades...")

    log = load_paper_log()

    print(
        f"Paper log rows: {len(log)}"
    )

    print("Loading BTC candles...")

    btc = load_btc()

    print(
        f"Latest BTC candle: "
        f"{btc['open_time'].iloc[-1]}"
    )

    required_columns = {
        "entry_price": None,
        "exit_price": None,
        "gross_return_pct": None,
        "fee_pct": None,
        "net_return_pct": None,
        "result": None,
        "status": None,
        "cumulative_net_return_pct": None,
    }

    for column, default_value in required_columns.items():
        if column not in log.columns:
            log[column] = default_value

    closed_count = 0

    for index, row in log.iterrows():
        signal = row["signal"]

        if signal not in [
            "BUY",
            "SELL",
        ]:
            log.at[
                index,
                "status",
            ] = "NO_TRADE"

            continue

        if (
            pd.notna(row.get("status"))
            and row.get("status") == "CLOSED"
        ):
            continue

        entry_timestamp = row["timestamp"]

        exit_timestamp = (
            entry_timestamp
            + pd.Timedelta(hours=1)
        )

        exit_match = btc[
            btc["open_time"]
            == exit_timestamp
        ]

        if exit_match.empty:
            log.at[
                index,
                "status",
            ] = "OPEN"

            continue

        entry_price = float(
            row["price"]
        )

        exit_price = float(
            exit_match.iloc[0]["close"]
        )

        if signal == "BUY":
            gross_return = (
                exit_price
                / entry_price
                - 1
            )

        else:
            gross_return = (
                entry_price
                / exit_price
                - 1
            )

        fee_decimal = (
            TRANSACTION_COST_PCT
            / 100
        )

        net_return = (
            gross_return
            - fee_decimal
        )

        gross_return_pct = (
            gross_return * 100
        )

        net_return_pct = (
            net_return * 100
        )

        result = (
            "WIN"
            if net_return > 0
            else "LOSS"
        )

        log.at[
            index,
            "entry_price",
        ] = entry_price

        log.at[
            index,
            "exit_price",
        ] = exit_price

        log.at[
            index,
            "gross_return_pct",
        ] = gross_return_pct

        log.at[
            index,
            "fee_pct",
        ] = TRANSACTION_COST_PCT

        log.at[
            index,
            "net_return_pct",
        ] = net_return_pct

        log.at[
            index,
            "result",
        ] = result

        log.at[
            index,
            "status",
        ] = "CLOSED"

        closed_count += 1

    # --------------------------------------------------------
    # Calculate cumulative net return for CLOSED trades only.
    # --------------------------------------------------------

    running_total = 0.0

    for index, row in log.iterrows():
        if (
            row["status"] == "CLOSED"
            and pd.notna(
                row["net_return_pct"]
            )
        ):
            running_total += float(
                row["net_return_pct"]
            )

        log.at[
            index,
            "cumulative_net_return_pct",
        ] = running_total

    log.to_csv(
        PAPER_LOG,
        index=False,
    )

    closed_trades = log[
        log["status"] == "CLOSED"
    ]

    wins = (
        closed_trades["result"]
        == "WIN"
    ).sum()

    losses = (
        closed_trades["result"]
        == "LOSS"
    ).sum()

    total_closed = len(
        closed_trades
    )

    win_rate = (
        wins
        / total_closed
        * 100
        if total_closed > 0
        else 0
    )

    total_net_return = (
        pd.to_numeric(
            closed_trades[
                "net_return_pct"
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    print("\n" + "=" * 70)
    print("PAPER TRADE SETTLEMENT")
    print("=" * 70)

    print(
        f"Trades settled this run: "
        f"{closed_count}"
    )

    print(
        f"Total closed trades:     "
        f"{total_closed}"
    )

    print(
        f"Wins:                    "
        f"{wins}"
    )

    print(
        f"Losses:                  "
        f"{losses}"
    )

    print(
        f"Win rate:                "
        f"{win_rate:.2f}%"
    )

    print(
        f"Total net return:        "
        f"{total_net_return:.2f}%"
    )

    print("=" * 70)

    print(
        f"\nUpdated paper log:\n"
        f"{PAPER_LOG}"
    )


if __name__ == "__main__":
    main()