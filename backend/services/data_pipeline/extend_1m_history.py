from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_URL = "https://data-api.binance.vision"
ENDPOINT = "/api/v3/klines"

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000

RAW_FILE = (
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


def parse_utc(
    value: str,
) -> pd.Timestamp:
    parsed = pd.Timestamp(value)

    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")

    return parsed


def to_milliseconds(
    value: pd.Timestamp,
) -> int:
    return int(
        value.timestamp() * 1000
    )


def fetch_page(
    session: requests.Session,
    start_ms: int,
    end_ms: int,
    retries: int = 6,
) -> list:
    for attempt in range(
        1,
        retries + 1,
    ):
        try:
            response = session.get(
                f"{BASE_URL}{ENDPOINT}",
                params={
                    "symbol": SYMBOL,
                    "interval": INTERVAL,
                    "limit": LIMIT,
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
                timeout=30,
            )

            if response.status_code in {
                418,
                429,
            }:
                wait_seconds = min(
                    60,
                    2 ** attempt,
                )

                print(
                    f"Rate limited "
                    f"({response.status_code}); "
                    f"waiting {wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:
            if attempt == retries:
                raise

            wait_seconds = min(
                30,
                2 ** attempt,
            )

            print(
                f"Request failed: {exc}. "
                f"Retrying in "
                f"{wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "Failed to fetch Binance page."
    )


def normalize(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
        errors="coerce",
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        column
        for column in COLUMNS
        if column not in {
            "open_time",
            "close_time",
        }
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    required = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df = df.dropna(
        subset=required
    )

    df = (
        df.drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    return df


def download_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    start_ms = to_milliseconds(
        start
    )

    end_ms = to_milliseconds(
        end
    )

    current = start_ms
    rows: list = []

    session = requests.Session()

    while current <= end_ms:
        page = fetch_page(
            session,
            current,
            end_ms,
        )

        if not page:
            break

        rows.extend(page)

        last_open_ms = int(
            page[-1][0]
        )

        next_start = (
            last_open_ms
            + 60_000
        )

        if next_start <= current:
            raise RuntimeError(
                "Binance pagination "
                "did not advance."
            )

        current = next_start

        if len(rows) % 100_000 < LIMIT:
            print(
                f"Downloaded "
                f"{len(rows):,} older candles "
                f"through "
                f"{pd.to_datetime(last_open_ms, unit='ms', utc=True)}"
            )

        time.sleep(0.08)

        if len(page) < LIMIT:
            break

    if not rows:
        raise RuntimeError(
            "No historical candles returned."
        )

    downloaded = pd.DataFrame(
        rows,
        columns=COLUMNS,
    )

    downloaded["open_time"] = (
        pd.to_datetime(
            downloaded["open_time"],
            unit="ms",
            utc=True,
        )
    )

    downloaded["close_time"] = (
        pd.to_datetime(
            downloaded["close_time"],
            unit="ms",
            utc=True,
        )
    )

    return normalize(
        downloaded
    )


def continuity_summary(
    df: pd.DataFrame,
) -> tuple[int, pd.DataFrame]:
    differences = (
        df["open_time"]
        .diff()
        .dropna()
    )

    bad = (
        differences
        != pd.Timedelta(minutes=1)
    )

    gap_count = int(
        bad.sum()
    )

    gap_rows = (
        df.loc[
            differences[
                bad
            ].index,
            ["open_time"],
        ]
        .head(10)
        .copy()
    )

    return (
        gap_count,
        gap_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extend the existing BTCUSDT "
            "1-minute dataset backward while "
            "preserving and validating the "
            "current data."
        )
    )

    parser.add_argument(
        "--start",
        default="2023-01-01",
        help=(
            "Earliest UTC candle to fetch. "
            "Default: 2023-01-01"
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_FILE,
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help=(
            "Skip creating a .bak copy of "
            "the existing raw CSV."
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Existing 1m file not found: "
            f"{args.input}"
        )

    print(
        "Loading existing 1-minute dataset..."
    )

    existing = pd.read_csv(
        args.input
    )

    existing = normalize(
        existing
    )

    existing_start = (
        existing["open_time"].iloc[0]
    )

    existing_end = (
        existing["open_time"].iloc[-1]
    )

    requested_start = parse_utc(
        args.start
    )

    print(
        f"Existing rows: "
        f"{len(existing):,}"
    )

    print(
        f"Existing start: "
        f"{existing_start}"
    )

    print(
        f"Existing end:   "
        f"{existing_end}"
    )

    if requested_start >= existing_start:
        print()
        print(
            "Nothing to extend. "
            "Requested start is not earlier "
            "than the current dataset."
        )
        return

    fetch_end = (
        existing_start
        - pd.Timedelta(minutes=1)
    )

    expected_older_rows = int(
        (
            fetch_end
            - requested_start
        )
        / pd.Timedelta(minutes=1)
    ) + 1

    print()
    print(
        f"Requested extended start: "
        f"{requested_start}"
    )

    print(
        f"Older range to fetch: "
        f"{requested_start} -> "
        f"{fetch_end}"
    )

    print(
        f"Expected older candles: "
        f"{expected_older_rows:,}"
    )

    print()
    print(
        "Downloading older Binance "
        "1-minute history..."
    )

    older = download_range(
        requested_start,
        fetch_end,
    )

    print()
    print(
        f"Downloaded older rows: "
        f"{len(older):,}"
    )

    combined = pd.concat(
        [
            older,
            existing,
        ],
        ignore_index=True,
    )

    combined = normalize(
        combined
    )

    gap_count, gap_rows = (
        continuity_summary(
            combined
        )
    )

    print()
    print(
        "Combined dataset validation:"
    )

    print(
        f"Rows:  {len(combined):,}"
    )

    print(
        f"Start: "
        f"{combined['open_time'].iloc[0]}"
    )

    print(
        f"End:   "
        f"{combined['open_time'].iloc[-1]}"
    )

    print(
        f"Gaps / irregular 1m intervals: "
        f"{gap_count}"
    )

    if gap_count:
        print()
        print(
            "First irregular timestamps:"
        )
        print(
            gap_rows.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Combined dataset is not "
            "continuous. Original file "
            "was not replaced."
        )

    if (
        combined[
            "open_time"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate timestamps remain "
            "after merge."
        )

    temp_file = (
        args.input.with_suffix(
            ".extended.tmp.csv"
        )
    )

    combined.to_csv(
        temp_file,
        index=False,
    )

    if not args.no_backup:
        backup_file = (
            args.input.with_suffix(
                ".pre_extension.bak.csv"
            )
        )

        print()
        print(
            f"Creating backup: "
            f"{backup_file}"
        )

        shutil.copy2(
            args.input,
            backup_file,
        )

    temp_file.replace(
        args.input
    )

    print()
    print(
        "Historical extension complete."
    )

    print(
        f"Updated raw file: "
        f"{args.input}"
    )

    print(
        f"Final rows: "
        f"{len(combined):,}"
    )

    print(
        f"Final start: "
        f"{combined['open_time'].iloc[0]}"
    )

    print(
        f"Final end: "
        f"{combined['open_time'].iloc[-1]}"
    )

    print(
        "Continuity check: PASSED"
    )


if __name__ == "__main__":
    main()
