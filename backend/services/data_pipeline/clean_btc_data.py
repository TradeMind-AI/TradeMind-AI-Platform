from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "btcusdt_1h.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "btcusdt_1h_clean.csv"


REQUIRED_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def clean_btc_data(df):
    original_rows = len(df)

    # Convert timestamps to UTC.
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        errors="coerce",
        utc=True,
    )

    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(
            df["close_time"],
            errors="coerce",
            utc=True,
        )

    # Convert market values to numeric.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove rows with invalid required timestamps or market values.
    df = df.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    rows_after_null_removal = len(df)

    # Remove duplicate candle timestamps.
    df = df.drop_duplicates(
        subset=["open_time"],
        keep="first",
    ).copy()

    rows_after_duplicates = len(df)

    # Remove invalid OHLC relationships.
    valid_ohlc = (
        (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["high"] >= df["low"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    )

    df = df[valid_ohlc].copy()

    rows_after_ohlc = len(df)

    # Remove negative volume.
    df = df[df["volume"] >= 0].copy()

    rows_after_volume = len(df)

    # Sort chronologically.
    df = df.sort_values(
        "open_time"
    ).reset_index(drop=True)

    # Detect missing or irregular hourly intervals.
    time_differences = (
        df["open_time"]
        .diff()
        .dropna()
    )

    missing_or_irregular_intervals = int(
        (time_differences != pd.Timedelta(hours=1)).sum()
    )

    print("\n=== BTC DATA CLEANING SUMMARY ===")

    print(f"Original rows: {original_rows}")
    print(
        f"Rows after null/invalid-value removal: "
        f"{rows_after_null_removal}"
    )
    print(
        f"Rows after duplicate removal: "
        f"{rows_after_duplicates}"
    )
    print(
        f"Rows after OHLC validation: "
        f"{rows_after_ohlc}"
    )
    print(
        f"Rows after volume validation: "
        f"{rows_after_volume}"
    )
    print(
        f"Final rows: {len(df)}"
    )
    print(
        f"Missing/irregular hourly intervals: "
        f"{missing_or_irregular_intervals}"
    )

    return df


def main():
    print("Loading raw BTC dataset...")
    print(f"Input: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    cleaned_df = clean_btc_data(df)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_df.to_csv(
        PROCESSED_FILE,
        index=False,
    )

    print("\nCleaning complete.")
    print(f"Output: {PROCESSED_FILE}")


if __name__ == "__main__":
    main()