from pathlib import Path

import pandas as pd
import requests


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BTC_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "btcusdt_1h.csv"
)

ETH_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ethusdt_1h.csv"
)

BNB_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bnbusdt_1h.csv"
)

DERIVATIVES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_derivatives_1h.csv"
)


# ============================================================
# Binance configuration
# ============================================================

SPOT_BASE_URL = "https://data-api.binance.vision"
KLINES_ENDPOINT = "/api/v3/klines"

FUTURES_BASE_URL = "https://fapi.binance.com"
FUNDING_ENDPOINT = "/fapi/v1/fundingRate"

INTERVAL = "1h"

# 200 hours is enough for incremental refresh.
KLINE_LIMIT = 200

# Binance allows up to 1000 funding records.
FUNDING_LIMIT = 1000


SYMBOL_FILES = {
    "BTCUSDT": BTC_FILE,
    "ETHUSDT": ETH_FILE,
    "BNBUSDT": BNB_FILE,
}


KLINE_COLUMNS = [
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


NUMERIC_KLINE_COLUMNS = [
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


# ============================================================
# Spot market data
# ============================================================

def fetch_latest_closed_klines(
    symbol: str,
) -> pd.DataFrame:
    """
    Download recent 1-hour candles and keep only candles
    that have completely closed.
    """

    response = requests.get(
        f"{SPOT_BASE_URL}{KLINES_ENDPOINT}",
        params={
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": KLINE_LIMIT,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        raise ValueError(
            f"No kline data returned for {symbol}."
        )

    df = pd.DataFrame(
        data,
        columns=KLINE_COLUMNS,
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

    for column in NUMERIC_KLINE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Never feed an unfinished candle into the ML model.
    # --------------------------------------------------------

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    df = df[
        df["close_time"] < now_utc
    ].copy()

    df = (
        df
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            f"No closed candles available for {symbol}."
        )

    return df


def update_spot_file(
    symbol: str,
):
    """
    Fetch recent closed candles and append/update them
    in the existing historical CSV without duplicates.
    """

    output_file = SYMBOL_FILES[symbol]

    print(
        f"\nFetching {symbol}..."
    )

    new_df = fetch_latest_closed_klines(
        symbol
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_file.exists():

        old_df = pd.read_csv(
            output_file
        )

        if not old_df.empty:

            old_df["open_time"] = pd.to_datetime(
                old_df["open_time"],
                utc=True,
            )

            old_df["close_time"] = pd.to_datetime(
                old_df["close_time"],
                utc=True,
            )

            combined = pd.concat(
                [
                    old_df,
                    new_df,
                ],
                ignore_index=True,
            )

        else:
            combined = new_df.copy()

    else:
        combined = new_df.copy()

    combined = (
        combined
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    combined.to_csv(
        output_file,
        index=False,
    )

    print(
        f"{symbol}: "
        f"{len(combined)} total rows"
    )

    print(
        "Latest closed candle: "
        f"{combined['open_time'].iloc[-1]}"
    )


# ============================================================
# Funding data
# ============================================================

def fetch_latest_funding() -> pd.DataFrame:
    """
    Download recent BTC perpetual funding-rate history.
    """

    response = requests.get(
        f"{FUTURES_BASE_URL}{FUNDING_ENDPOINT}",
        params={
            "symbol": "BTCUSDT",
            "limit": FUNDING_LIMIT,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        raise ValueError(
            "No BTC funding data returned."
        )

    df = pd.DataFrame(data)

    required_columns = [
        "fundingTime",
        "fundingRate",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Funding API response is missing columns: "
            f"{missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["fundingTime"],
        unit="ms",
        utc=True,
    )

    df["funding_rate"] = pd.to_numeric(
        df["fundingRate"],
        errors="coerce",
    )

    df = df[
        [
            "timestamp",
            "funding_rate",
        ]
    ].copy()

    df = df.dropna(
        subset=[
            "timestamp",
            "funding_rate",
        ]
    )

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def convert_funding_to_hourly(
    funding: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert funding events into an hourly funding feature.

    Funding normally updates every several hours.
    Between events we forward-fill the last known value.
    """

    funding = funding.copy()

    funding["timestamp"] = (
        funding["timestamp"]
        .dt.floor("h")
    )

    # In the unlikely event multiple funding records map
    # to one hour, keep the most recent one.
    funding = (
        funding
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    start_time = (
        funding["timestamp"]
        .min()
    )

    end_time = (
        funding["timestamp"]
        .max()
    )

    hourly_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq="1h",
        tz="UTC",
    )

    hourly = pd.DataFrame(
        {
            "timestamp": hourly_index
        }
    )

    hourly = hourly.merge(
        funding,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    hourly["funding_rate"] = (
        hourly["funding_rate"]
        .ffill()
    )

    return hourly


def load_existing_derivatives() -> pd.DataFrame:
    """
    Load the existing historical derivatives CSV.

    Your historical file may use:
        timestamp

    Older/newer experiments may use:
        open_time

    Normalize both to timestamp.
    """

    if not DERIVATIVES_FILE.exists():

        return pd.DataFrame(
            columns=[
                "timestamp",
                "funding_rate",
            ]
        )

    old = pd.read_csv(
        DERIVATIVES_FILE
    )

    if old.empty:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "funding_rate",
            ]
        )

    if "timestamp" in old.columns:

        pass

    elif "open_time" in old.columns:

        old = old.rename(
            columns={
                "open_time": "timestamp"
            }
        )

    else:

        raise ValueError(
            "\nExisting derivatives file does not "
            "contain a timestamp column.\n"
            f"File: {DERIVATIVES_FILE}\n"
            f"Columns found: {list(old.columns)}"
        )

    if "funding_rate" not in old.columns:

        raise ValueError(
            "\nExisting derivatives file does not "
            "contain 'funding_rate'.\n"
            f"Columns found: {list(old.columns)}"
        )

    old["timestamp"] = pd.to_datetime(
        old["timestamp"],
        utc=True,
    )

    old["funding_rate"] = pd.to_numeric(
        old["funding_rate"],
        errors="coerce",
    )

    old = old[
        [
            "timestamp",
            "funding_rate",
        ]
    ].copy()

    old = old.dropna(
        subset=["timestamp"]
    )

    old = (
        old
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return old


def update_derivatives():
    """
    Refresh BTC funding-rate data while preserving the
    historical derivatives dataset.
    """

    print(
        "\nFetching BTC funding data..."
    )

    funding_events = (
        fetch_latest_funding()
    )

    print(
        "Funding events fetched: "
        f"{len(funding_events)}"
    )

    new_hourly = (
        convert_funding_to_hourly(
            funding_events
        )
    )

    old = (
        load_existing_derivatives()
    )

    print(
        "Existing derivatives rows: "
        f"{len(old)}"
    )

    combined = pd.concat(
        [
            old,
            new_hourly,
        ],
        ignore_index=True,
    )

    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"],
        utc=True,
    )

    combined["funding_rate"] = pd.to_numeric(
        combined["funding_rate"],
        errors="coerce",
    )

    combined = (
        combined
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Forward fill only.
    #
    # Do NOT backward fill because that could introduce
    # information from a future funding observation.
    # --------------------------------------------------------

    combined["funding_rate"] = (
        combined["funding_rate"]
        .ffill()
    )

    DERIVATIVES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Save using "timestamp" because this matches the existing
    # historical derivatives dataset format.
    # --------------------------------------------------------

    combined.to_csv(
        DERIVATIVES_FILE,
        index=False,
    )

    print(
        "Funding rows: "
        f"{len(combined)}"
    )

    print(
        "Latest funding hour: "
        f"{combined['timestamp'].iloc[-1]}"
    )

    print(
        "Funding file: "
        f"{DERIVATIVES_FILE}"
    )


# ============================================================
# Validation
# ============================================================

def validate_latest_spot_timestamps():
    """
    Verify BTC, ETH and BNB end at the same candle.
    """

    latest_times = {}

    for symbol, path in SYMBOL_FILES.items():

        df = pd.read_csv(
            path,
            usecols=["open_time"],
        )

        df["open_time"] = pd.to_datetime(
            df["open_time"],
            utc=True,
        )

        latest_times[symbol] = (
            df["open_time"].max()
        )

    print(
        "\nLatest spot timestamps:"
    )

    for symbol, timestamp in latest_times.items():

        print(
            f"{symbol}: {timestamp}"
        )

    unique_times = set(
        latest_times.values()
    )

    if len(unique_times) == 1:

        print(
            "Spot timestamp alignment: OK"
        )

    else:

        print(
            "WARNING: spot datasets do not "
            "end at the same timestamp."
        )


# ============================================================
# Main
# ============================================================

def main():
    print(
        "=" * 70
    )

    print(
        "UPDATING LIVE MARKET DATA"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # BTC
    # --------------------------------------------------------

    update_spot_file(
        "BTCUSDT"
    )

    # --------------------------------------------------------
    # ETH
    # --------------------------------------------------------

    update_spot_file(
        "ETHUSDT"
    )

    # --------------------------------------------------------
    # BNB
    # --------------------------------------------------------

    update_spot_file(
        "BNBUSDT"
    )

    # --------------------------------------------------------
    # Funding
    # --------------------------------------------------------

    update_derivatives()

    # --------------------------------------------------------
    # Alignment check
    # --------------------------------------------------------

    validate_latest_spot_timestamps()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LIVE MARKET DATA UPDATE COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()