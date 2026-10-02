from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bnbusdt_1h.csv"
)

BASE_URL = "https://data-api.binance.vision"
ENDPOINT = "/api/v3/klines"

SYMBOL = "BNBUSDT"
INTERVAL = "1h"
LIMIT = 1000

START_DATE = "2023-09-25"
END_DATE = "2026-09-25"


def to_milliseconds(date_string: str) -> int:
    dt = datetime.strptime(
        date_string,
        "%Y-%m-%d",
    ).replace(
        tzinfo=timezone.utc,
    )

    return int(dt.timestamp() * 1000)


def fetch_klines(start_ms: int, end_ms: int) -> list:
    rows = []
    current_start = start_ms

    print("Fetching BNB/USDT 1h history...")

    while current_start < end_ms:
        response = requests.get(
            f"{BASE_URL}{ENDPOINT}",
            params={
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": LIMIT,
            },
            timeout=20,
        )

        response.raise_for_status()
        data = response.json()

        if not data:
            break

        rows.extend(data)

        last_open_time = int(data[-1][0])
        current_start = last_open_time + 1

        print(f"Rows collected: {len(rows)}")

        if len(data) < LIMIT:
            break

    return rows


def main():
    rows = fetch_klines(
        to_milliseconds(START_DATE),
        to_milliseconds(END_DATE),
    )

    if not rows:
        raise ValueError("No BNB data returned.")

    columns = [
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

    df = pd.DataFrame(rows, columns=columns)

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
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nBNB history download complete.")
    print(f"Rows: {len(df)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()