from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_derivatives_1h.csv"
)

BASE_URL = "https://fapi.binance.com"

SYMBOL = "BTCUSDT"

START_DATE = "2023-09-25"
END_DATE = "2026-09-24"


def to_milliseconds(date_string: str) -> int:
    dt = datetime.strptime(
        date_string,
        "%Y-%m-%d",
    ).replace(
        tzinfo=timezone.utc,
    )

    return int(dt.timestamp() * 1000)


def fetch_funding_rates(
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    print("Fetching funding rates...")

    rows = []

    current_start = start_ms

    while current_start < end_ms:
        response = requests.get(
            f"{BASE_URL}/fapi/v1/fundingRate",
            params={
                "symbol": SYMBOL,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            break

        rows.extend(data)

        last_time = int(
            data[-1]["fundingTime"]
        )

        current_start = last_time + 1

        print(
            f"Funding rows collected: {len(rows)}"
        )

        if len(data) < 1000:
            break

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["fundingTime"],
        unit="ms",
        utc=True,
    )

    df["funding_rate"] = pd.to_numeric(
        df["fundingRate"],
        errors="coerce",
    )

    return df[
        [
            "timestamp",
            "funding_rate",
        ]
    ].drop_duplicates(
        subset=["timestamp"]
    )


def fetch_open_interest(
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    print("\nFetching open interest history...")

    rows = []

    current_start = start_ms

    while current_start < end_ms:
        response = requests.get(
            f"{BASE_URL}/futures/data/openInterestHist",
            params={
                "symbol": SYMBOL,
                "period": "1h",
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 500,
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            break

        rows.extend(data)

        last_time = int(
            data[-1]["timestamp"]
        )

        current_start = last_time + 1

        print(
            f"Open interest rows collected: {len(rows)}"
        )

        if len(data) < 500:
            break

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df["open_interest"] = pd.to_numeric(
        df["sumOpenInterest"],
        errors="coerce",
    )

    df["open_interest_value"] = pd.to_numeric(
        df["sumOpenInterestValue"],
        errors="coerce",
    )

    return df[
        [
            "timestamp",
            "open_interest",
            "open_interest_value",
        ]
    ].drop_duplicates(
        subset=["timestamp"]
    )


def main():
    start_ms = to_milliseconds(START_DATE)
    end_ms = to_milliseconds(END_DATE)

    funding_df = fetch_funding_rates(
        start_ms,
        end_ms,
    )

    print("\nFunding rows:")
    print(len(funding_df))

    if funding_df.empty:
        raise ValueError(
            "No funding-rate data returned."
        )

    funding_hourly = (
        funding_df
        .set_index("timestamp")
        .resample("1h")
        .ffill()
        .reset_index()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    funding_hourly.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nHistorical derivatives data saved.")
    print(f"Rows: {len(funding_hourly)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nFirst 5 rows:")
    print(
        funding_hourly.head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()