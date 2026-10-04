from pathlib import Path
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "btcusdt_1m.csv"
)

BASE_URL = "https://data-api.binance.vision"
ENDPOINT = "/api/v3/klines"
LIMIT = 1000

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


def fetch_page(
    start_ms: int,
    end_ms: int,
) -> list:
    response = requests.get(
        f"{BASE_URL}{ENDPOINT}",
        params={
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": LIMIT,
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"1m raw file not found: "
            f"{RAW_FILE}"
        )

    timestamps = pd.read_csv(
        RAW_FILE,
        usecols=["open_time"],
    )

    timestamps["open_time"] = (
        pd.to_datetime(
            timestamps["open_time"],
            utc=True,
        )
    )

    latest = timestamps[
        "open_time"
    ].max()

    now = pd.Timestamp.now(
        tz="UTC"
    )

    # Only request candles that have fully closed.
    latest_closed_open = (
        now.floor("min")
        - pd.Timedelta(minutes=1)
    )

    start = (
        latest
        + pd.Timedelta(minutes=1)
    )

    print(
        f"Existing latest 1m candle: "
        f"{latest}"
    )

    if start > latest_closed_open:
        print(
            "1m data already up to date."
        )
        return

    rows = []
    current_ms = int(
        start.timestamp() * 1000
    )

    end_ms = int(
        latest_closed_open.timestamp()
        * 1000
    )

    while current_ms <= end_ms:
        page = fetch_page(
            current_ms,
            end_ms,
        )

        if not page:
            break

        rows.extend(page)

        last_open_ms = int(
            page[-1][0]
        )

        next_ms = (
            last_open_ms
            + 60_000
        )

        if next_ms <= current_ms:
            raise RuntimeError(
                "Binance pagination "
                "did not advance."
            )

        current_ms = next_ms

        time.sleep(0.05)

        if len(page) < LIMIT:
            break

    if not rows:
        print(
            "No newer closed 1m candles "
            "were returned."
        )
        return

    new = pd.DataFrame(
        rows,
        columns=COLUMNS,
    )

    new["open_time"] = pd.to_datetime(
        new["open_time"],
        unit="ms",
        utc=True,
    )

    new["close_time"] = pd.to_datetime(
        new["close_time"],
        unit="ms",
        utc=True,
    )

    new = (
        new[
            new["open_time"] > latest
        ]
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    if new.empty:
        print(
            "No new rows after duplicate "
            "filtering."
        )
        return

    first_new = new[
        "open_time"
    ].iloc[0]

    expected_first = (
        latest
        + pd.Timedelta(minutes=1)
    )

    if first_new != expected_first:
        print(
            "WARNING: live source contains "
            "a missing 1m interval."
        )

        print(
            f"Expected: {expected_first}"
        )

        print(
            f"Received: {first_new}"
        )

        print(
            "The gap will be preserved; "
            "no synthetic candles are added."
        )

    new.to_csv(
        RAW_FILE,
        mode="a",
        header=False,
        index=False,
    )

    print(
        f"Appended {len(new):,} "
        f"closed 1m candles."
    )

    print(
        f"New latest candle: "
        f"{new['open_time'].iloc[-1]}"
    )


if __name__ == "__main__":
    main()
