from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_URL = "https://data-api.binance.vision"
ENDPOINT = "/api/v3/klines"

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "btcusdt_1m.csv"
)


COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def parse_utc(value: str) -> datetime:
    parsed = pd.Timestamp(value)

    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")

    return parsed.to_pydatetime()


def to_milliseconds(value: datetime) -> int:
    return int(
        value.timestamp() * 1000
    )


def fetch_page(
    start_ms: int,
    end_ms: int,
) -> list:
    response = requests.get(
        f"{BASE_URL}{ENDPOINT}",
        params={
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "limit": LIMIT,
            "startTime": start_ms,
            "endTime": end_ms,
        },
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download historical BTCUSDT "
            "1-minute Binance spot candles."
        )
    )

    parser.add_argument(
        "--start",
        required=True,
        help="UTC start, e.g. 2026-01-01",
    )

    parser.add_argument(
        "--end",
        default=None,
        help=(
            "UTC end. Defaults to current UTC time."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
    )

    args = parser.parse_args()

    start = parse_utc(
        args.start
    )

    end = (
        parse_utc(args.end)
        if args.end
        else datetime.now(
            timezone.utc
        )
    )

    if start >= end:
        raise ValueError(
            "Start time must be before end time."
        )

    start_ms = to_milliseconds(
        start
    )

    end_ms = to_milliseconds(
        end
    )

    rows: list = []

    current = start_ms

    while current < end_ms:
        page = fetch_page(
            current,
            end_ms,
        )

        if not page:
            break

        rows.extend(page)

        last_open_time = int(
            page[-1][0]
        )

        next_start = (
            last_open_time
            + 60_000
        )

        if next_start <= current:
            raise RuntimeError(
                "Binance pagination did not advance."
            )

        current = next_start

        print(
            f"Downloaded {len(rows):,} candles "
            f"through "
            f"{pd.to_datetime(last_open_time, unit='ms', utc=True)}"
        )

        time.sleep(0.08)

        if len(page) < LIMIT:
            break

    if not rows:
        raise RuntimeError(
            "No candles were returned by Binance."
        )

    df = pd.DataFrame(
        rows,
        columns=COLUMNS,
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True,
    )

    numeric_columns = [
        column
        for column in COLUMNS
        if column
        not in {
            "open_time",
            "close_time",
        }
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        args.output,
        index=False,
    )

    print(
        f"Saved {len(df):,} candles to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
