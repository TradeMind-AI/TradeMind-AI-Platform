import requests
import pandas as pd


BASE_URL = "https://data-api.binance.vision"
ENDPOINT = "/api/v3/klines"

SYMBOL = "BTCUSDT"
INTERVAL = "1h"


def fetch_historical_batch(start_time, limit=1000):
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_time,
        "limit": limit,
    }

    response = requests.get(
        f"{BASE_URL}{ENDPOINT}",
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

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

    df = pd.DataFrame(data, columns=columns)

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
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    return df


def fetch_historical_data(start_time, end_time, batch_size=1000):
    all_batches = []

    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    if start_time.tzinfo is None:
        start_time = start_time.tz_localize("UTC")
    else:
        start_time = start_time.tz_convert("UTC")

    if end_time.tzinfo is None:
        end_time = end_time.tz_localize("UTC")
    else:
        end_time = end_time.tz_convert("UTC")

    current_start = start_time

    while current_start < end_time:
        print(f"Fetching batch starting at {current_start}...")

        start_time_ms = int(current_start.timestamp() * 1000)

        df = fetch_historical_batch(
            start_time=start_time_ms,
            limit=batch_size,
        )

        if df.empty:
            break

        df = df[df["open_time"] < end_time]

        if df.empty:
            break

        all_batches.append(df)

        current_start = (
         df["open_time"].iloc[-1]
         + pd.Timedelta(hours=1)
        )

        if len(df) < batch_size:
            break

    if not all_batches:
        return pd.DataFrame()

    historical_df = pd.concat(
        all_batches,
        ignore_index=True,
    )

    historical_df = historical_df.drop_duplicates(
        subset=["open_time"]
    )

    historical_df = historical_df.sort_values(
        "open_time"
    ).reset_index(drop=True)

    return historical_df