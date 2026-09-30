from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "btcusdt_1h_clean.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "btcusdt_1h_features.csv"


def load_clean_data() -> pd.DataFrame:
    """Load the validated and cleaned BTC dataset."""

    df = pd.read_csv(INPUT_FILE)

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)

    return df


def create_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create historical return and log-return features."""

    df["return_1h"] = df["close"].pct_change(1)
    df["return_3h"] = df["close"].pct_change(3)
    df["return_6h"] = df["close"].pct_change(6)
    df["return_12h"] = df["close"].pct_change(12)
    df["return_24h"] = df["close"].pct_change(24)

    df["log_return_1h"] = np.log(
        df["close"] / df["close"].shift(1)
    )

    return df


def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create moving-average and price-vs-trend features."""

    df["sma_6"] = df["close"].rolling(window=6).mean()
    df["sma_12"] = df["close"].rolling(window=12).mean()
    df["sma_24"] = df["close"].rolling(window=24).mean()
    df["sma_48"] = df["close"].rolling(window=48).mean()
    df["sma_168"] = df["close"].rolling(window=168).mean()

    df["ema_12"] = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    df["ema_24"] = df["close"].ewm(
        span=24,
        adjust=False
    ).mean()

    df["ema_50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["close_to_sma_24"] = (
        df["close"] / df["sma_24"] - 1
    )

    df["close_to_sma_168"] = (
        df["close"] / df["sma_168"] - 1
    )

    return df


def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create RSI and MACD momentum features."""

    price_change = df["close"].diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()

    relative_strength = average_gain / average_loss

    df["rsi_14"] = 100 - (
        100 / (1 + relative_strength)
    )

    ema_12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema_12 - ema_26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_histogram"] = (
        df["macd"] - df["macd_signal"]
    )

    return df


def create_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling volatility, ATR, and Bollinger Band features."""

    df["rolling_volatility_24"] = (
        df["return_1h"].rolling(window=24).std()
    )

    df["rolling_volatility_72"] = (
        df["return_1h"].rolling(window=72).std()
    )

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = true_range.rolling(
        window=14
    ).mean()

    bollinger_middle = (
        df["close"].rolling(window=20).mean()
    )

    bollinger_std = (
        df["close"].rolling(window=20).std()
    )

    df["bollinger_middle_20"] = bollinger_middle

    df["bollinger_upper_20"] = (
        bollinger_middle + (2 * bollinger_std)
    )

    df["bollinger_lower_20"] = (
        bollinger_middle - (2 * bollinger_std)
    )

    df["bollinger_width_20"] = (
        (
            df["bollinger_upper_20"]
            - df["bollinger_lower_20"]
        )
        / df["bollinger_middle_20"]
    )

    return df


def create_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create candle-structure features."""

    df["candle_range"] = (
        df["high"] - df["low"]
    )

    df["body_size"] = (
        df["close"] - df["open"]
    ).abs()

    df["upper_wick"] = (
        df["high"]
        - df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    )

    df["range_pct"] = (
        df["candle_range"] / df["open"]
    )

    df["body_pct"] = (
        df["body_size"] / df["open"]
    )

    return df


def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create volume and trading-activity features."""

    df["volume_change"] = (
        df["volume"].pct_change()
    )

    df["volume_sma_24"] = (
        df["volume"].rolling(window=24).mean()
    )

    df["relative_volume"] = (
        df["volume"] / df["volume_sma_24"]
    )

    df["trade_count_change"] = (
        df["number_of_trades"].pct_change()
    )

    df["trade_count_sma_24"] = (
        df["number_of_trades"]
        .rolling(window=24)
        .mean()
    )

    df["taker_buy_ratio"] = (
        df["taker_buy_base_volume"]
        / df["volume"]
    )

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and cyclical time features."""

    df["hour"] = df["open_time"].dt.hour
    df["day_of_week"] = df["open_time"].dt.dayofweek
    df["day_of_month"] = df["open_time"].dt.day
    df["month"] = df["open_time"].dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged features using only historical observations."""

    df["return_1h_lag_1"] = df["return_1h"].shift(1)
    df["return_1h_lag_2"] = df["return_1h"].shift(2)
    df["return_1h_lag_3"] = df["return_1h"].shift(3)
    df["return_1h_lag_6"] = df["return_1h"].shift(6)
    df["return_1h_lag_12"] = df["return_1h"].shift(12)
    df["return_1h_lag_24"] = df["return_1h"].shift(24)

    df["volume_lag_1"] = df["volume"].shift(1)
    df["volume_lag_3"] = df["volume"].shift(3)
    df["volume_lag_6"] = df["volume"].shift(6)
    df["volume_lag_24"] = df["volume"].shift(24)

    df["rsi_14_lag_1"] = df["rsi_14"].shift(1)
    df["rsi_14_lag_3"] = df["rsi_14"].shift(3)
    df["rsi_14_lag_6"] = df["rsi_14"].shift(6)

    df["volatility_24_lag_1"] = (
        df["rolling_volatility_24"].shift(1)
    )

    df["volatility_24_lag_3"] = (
        df["rolling_volatility_24"].shift(3)
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all feature groups."""

    df = create_return_features(df)
    df = create_trend_features(df)
    df = create_momentum_features(df)
    df = create_volatility_features(df)
    df = create_candle_features(df)
    df = create_volume_features(df)
    df = create_time_features(df)
    df = create_lag_features(df)

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the 1-hour-ahead return and three-class target.

    Target:
        DOWN    < -0.001
        NEUTRAL  -0.001 to +0.001
        UP       > +0.001
    """

    # Future return from the current close to the next hourly close.
    df["future_return_1h"] = (
        df["close"].shift(-1) / df["close"] - 1
    )

    # Three-class classification target.
    df["target"] = np.select(
        [
            df["future_return_1h"] < -0.001,
            df["future_return_1h"] > 0.001,
        ],
        [
            "DOWN",
            "UP",
        ],
        default="NEUTRAL",
    )

    return df


def remove_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows affected by feature warm-up periods."""

    warmup_period = 167

    print(
        f"\nRemoving first {warmup_period} warm-up rows..."
    )

    df = df.iloc[warmup_period:].copy()

    print(
        f"Rows after warm-up removal: {len(df)}"
    )

    return df


def main() -> None:
    """Run the complete feature and target preparation pipeline."""

    print("Loading cleaned BTC dataset...")

    df = load_clean_data()

    print(f"Input rows: {len(df)}")

    print("\nCreating features...")

    df = create_features(df)

    print("Feature creation complete.")

    # Remove initial rows where rolling/lag features
    # do not have sufficient historical observations.
    df = remove_warmup_rows(df)

    # Create future target after feature creation.
    df = create_target(df)

    # The final row has no known next-hour close,
    # therefore its target cannot be calculated.
    before_target_cleanup = len(df)

    df = df.dropna(
        subset=["future_return_1h", "target"]
    ).copy()

    removed_target_rows = (
        before_target_cleanup - len(df)
    )

    print(
        f"Rows removed because future target is unavailable: "
        f"{removed_target_rows}"
    )

    print(
        f"Final ML-ready rows: {len(df)}"
    )

    # Feature list
    original_columns = {
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
    }

    target_columns = {
        "future_return_1h",
        "target",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in original_columns
        and column not in target_columns
    ]

    print(
        f"\nTotal engineered features: "
        f"{len(feature_columns)}"
    )

    print("\nTarget distribution:")

    print(
        df["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTarget percentages:")

    print(
        (
            df["target"]
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
            .astype(str)
            + "%"
        ).to_string()
    )

    print("\nFuture return statistics:")

    print(
        df["future_return_1h"]
        .describe()
        .to_string()
    )

    print("\nRemaining NaN values:")

    remaining_nan = (
        df[feature_columns + ["future_return_1h", "target"]]
        .isna()
        .sum()
    )

    remaining_nan = remaining_nan[
        remaining_nan > 0
    ]

    if remaining_nan.empty:
        print("No NaN values remain.")
    else:
        print(remaining_nan.to_string())

    print("\nSaving ML-ready feature dataset...")

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Feature dataset saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()