import pandas as pd


def validate_btc_data(df):
    print("\n=== BTC DATA VALIDATION ===")

    # 1. Check for duplicate timestamps
    duplicate_count = df["open_time"].duplicated().sum()

    print(f"Duplicate timestamps: {duplicate_count}")

    # 2. Check timestamp ordering
    is_sorted = df["open_time"].is_monotonic_increasing

    print(f"Timestamps sorted: {is_sorted}")

    # 3. Check for missing hourly candles
    time_differences = df["open_time"].diff().dropna()

    expected_interval = pd.Timedelta(hours=1)

    missing_intervals = (
        time_differences != expected_interval
    ).sum()

    print(f"Missing/irregular hourly intervals: {missing_intervals}")

    # 4. Check OHLC validity
    invalid_ohlc = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ).sum()

    print(f"Invalid OHLC rows: {invalid_ohlc}")

    # 5. Check for negative volume
    negative_volume = (
        df["volume"] < 0
    ).sum()

    print(f"Negative volume rows: {negative_volume}")

    # 6. Check for null values
    null_values = df.isnull().sum().sum()

    print(f"Total null values: {null_values}")

    # Overall result
    validation_passed = (
        duplicate_count == 0
        and is_sorted
        and missing_intervals == 0
        and invalid_ohlc == 0
        and negative_volume == 0
        and null_values == 0
    )

    print(
        f"\nValidation passed: {validation_passed}"
    )

    return validation_passed