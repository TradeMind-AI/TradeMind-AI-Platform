from pathlib import Path

import pandas as pd

from risk_manager import STARTING_BALANCE


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


# Keep this consistent with our historical comparison.
TRANSACTION_COST_PCT = 0.05


def load_log():
    if not PAPER_LOG.exists():
        raise FileNotFoundError(
            f"Paper log not found:\n{PAPER_LOG}"
        )

    log = pd.read_csv(
        PAPER_LOG
    )

    log["timestamp"] = pd.to_datetime(
        log["timestamp"],
        utc=True,
    )

    return log


def load_btc():
    btc = pd.read_csv(
        BTC_FILE
    )

    btc["open_time"] = pd.to_datetime(
        btc["open_time"],
        utc=True,
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:
        btc[column] = pd.to_numeric(
            btc[column],
            errors="coerce",
        )

    return (
        btc
        .sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def main():

    print("Loading paper trades...")

    log = load_log()

    print(
        f"Paper log rows: {len(log)}"
    )

    print("Loading BTC candles...")

    btc = load_btc()

    print(
        f"Latest BTC candle: "
        f"{btc['open_time'].iloc[-1]}"
    )

    required = [
        "entry_price",
        "exit_price",
        "exit_reason",
        "gross_return_pct",
        "fee_pct",
        "net_return_pct",
        "account_pnl",
        "account_balance_after",
        "result",
        "status",
        "cumulative_net_return_pct",
    ]

    for column in required:
        if column not in log.columns:
            log[column] = pd.NA

    balance = STARTING_BALANCE

    existing_pnl = pd.to_numeric(
        log["account_pnl"],
        errors="coerce",
    ).fillna(0)

    balance += existing_pnl.sum()

    newly_closed = 0

    for index, row in log.iterrows():

        signal = row["signal"]

        if signal not in {
            "BUY",
            "SELL",
        }:
            log.at[
                index,
                "status",
            ] = "NO_TRADE"

            continue

        if row["status"] == "CLOSED":
            continue

        entry_timestamp = row[
            "timestamp"
        ]

        next_timestamp = (
            entry_timestamp
            + pd.Timedelta(hours=1)
        )

        candle_match = btc[
            btc["open_time"]
            == next_timestamp
        ]

        if candle_match.empty:

            log.at[
                index,
                "status",
            ] = "OPEN"

            continue

        candle = candle_match.iloc[0]

        entry_price = float(
            row["entry_price"]
            if pd.notna(
                row["entry_price"]
            )
            else row["price"]
        )

        stop_price = float(
            row["stop_loss_price"]
        )

        target_price = float(
            row["take_profit_price"]
        )

        high = float(
            candle["high"]
        )

        low = float(
            candle["low"]
        )

        close = float(
            candle["close"]
        )

        # ----------------------------------------------------
        # Determine whether stop or target was reached.
        #
        # If both are touched in the same 1h candle,
        # intrabar ordering is unknown.
        #
        # Conservative assumption:
        # STOP LOSS happened first.
        # ----------------------------------------------------

        if signal == "BUY":

            stop_hit = (
                low <= stop_price
            )

            target_hit = (
                high >= target_price
            )

        else:

            stop_hit = (
                high >= stop_price
            )

            target_hit = (
                low <= target_price
            )

        if stop_hit:

            exit_price = stop_price
            exit_reason = "STOP_LOSS"

        elif target_hit:

            exit_price = target_price
            exit_reason = "TAKE_PROFIT"

        else:

            exit_price = close
            exit_reason = "ONE_HOUR_EXIT"

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

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

        position_notional = float(
            row["position_notional"]
        )

        account_pnl = (
            position_notional
            * net_return
        )

        balance += account_pnl

        result = (
            "WIN"
            if account_pnl > 0
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
            "exit_reason",
        ] = exit_reason

        log.at[
            index,
            "gross_return_pct",
        ] = (
            gross_return * 100
        )

        log.at[
            index,
            "fee_pct",
        ] = TRANSACTION_COST_PCT

        log.at[
            index,
            "net_return_pct",
        ] = (
            net_return * 100
        )

        log.at[
            index,
            "account_pnl",
        ] = account_pnl

        log.at[
            index,
            "account_balance_after",
        ] = balance

        log.at[
            index,
            "result",
        ] = result

        log.at[
            index,
            "status",
        ] = "CLOSED"

        newly_closed += 1

    # --------------------------------------------------------
    # Recalculate cumulative returns in chronological order.
    # --------------------------------------------------------

    cumulative = 0.0

    for index, row in log.iterrows():

        if (
            row["status"] == "CLOSED"
            and pd.notna(
                row["net_return_pct"]
            )
        ):

            cumulative += float(
                row["net_return_pct"]
            )

        log.at[
            index,
            "cumulative_net_return_pct",
        ] = cumulative

    log.to_csv(
        PAPER_LOG,
        index=False,
    )

    closed = log[
        log["status"] == "CLOSED"
    ].copy()

    wins = (
        closed["result"] == "WIN"
    ).sum()

    losses = (
        closed["result"] == "LOSS"
    ).sum()

    total = len(
        closed
    )

    win_rate = (
        wins / total * 100
        if total
        else 0.0
    )

    total_account_pnl = pd.to_numeric(
        closed["account_pnl"],
        errors="coerce",
    ).fillna(0).sum()

    ending_balance = (
        STARTING_BALANCE
        + total_account_pnl
    )

    print("\n" + "=" * 72)
    print("PAPER TRADE SETTLEMENT")
    print("=" * 72)

    print(
        f"Trades settled this run: "
        f"{newly_closed}"
    )

    print(
        f"Total closed trades:     "
        f"{total}"
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
        f"Account P&L:             "
        f"${total_account_pnl:.2f}"
    )

    print(
        f"Paper balance:           "
        f"${ending_balance:.2f}"
    )

    print("=" * 72)

    print(
        f"\nUpdated paper log:\n"
        f"{PAPER_LOG}"
    )


if __name__ == "__main__":
    main()