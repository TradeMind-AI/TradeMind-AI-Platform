from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1m.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1m_features.csv"
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        utc=True,
    )

    df = (
        df.sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def create_return_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    horizons = [
        1,
        3,
        5,
        10,
        20,
        30,
        60,
    ]

    for horizon in horizons:
        df[
            f"return_{horizon}m"
        ] = (
            df["close"]
            .pct_change(horizon)
        )

    df["log_return_1m"] = np.log(
        df["close"]
        / df["close"].shift(1)
    )

    return df


def create_trend_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    windows = [
        5,
        10,
        20,
        30,
        60,
        120,
        240,
    ]

    for window in windows:
        df[
            f"sma_{window}"
        ] = (
            df["close"]
            .rolling(window)
            .mean()
        )

    for span in [
        12,
        26,
        50,
        100,
    ]:
        df[
            f"ema_{span}"
        ] = (
            df["close"]
            .ewm(
                span=span,
                adjust=False,
            )
            .mean()
        )

    df["close_to_sma_20"] = (
        df["close"]
        / df["sma_20"]
        - 1
    )

    df["close_to_sma_60"] = (
        df["close"]
        / df["sma_60"]
        - 1
    )

    df["close_to_sma_240"] = (
        df["close"]
        / df["sma_240"]
        - 1
    )

    return df


def create_momentum_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    price_change = (
        df["close"].diff()
    )

    gain = (
        price_change.clip(
            lower=0
        )
    )

    loss = (
        -price_change.clip(
            upper=0
        )
    )

    average_gain = (
        gain
        .rolling(14)
        .mean()
    )

    average_loss = (
        loss
        .rolling(14)
        .mean()
    )

    relative_strength = (
        average_gain
        / average_loss
    )

    df["rsi_14"] = (
        100
        - (
            100
            / (
                1
                + relative_strength
            )
        )
    )

    ema_12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema_26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    df["macd"] = (
        ema_12
        - ema_26
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["macd_histogram"] = (
        df["macd"]
        - df["macd_signal"]
    )

    return df


def create_volatility_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    for window in [
        10,
        30,
        60,
        240,
    ]:
        df[
            f"rolling_volatility_{window}"
        ] = (
            df["return_1m"]
            .rolling(window)
            .std()
        )

    previous_close = (
        df["close"].shift(1)
    )

    true_range = pd.concat(
        [
            df["high"]
            - df["low"],
            (
                df["high"]
                - previous_close
            ).abs(),
            (
                df["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = (
        true_range
        .rolling(14)
        .mean()
    )

    bollinger_middle = (
        df["close"]
        .rolling(20)
        .mean()
    )

    bollinger_std = (
        df["close"]
        .rolling(20)
        .std()
    )

    df["bollinger_middle_20"] = (
        bollinger_middle
    )

    df["bollinger_upper_20"] = (
        bollinger_middle
        + 2 * bollinger_std
    )

    df["bollinger_lower_20"] = (
        bollinger_middle
        - 2 * bollinger_std
    )

    df["bollinger_width_20"] = (
        (
            df["bollinger_upper_20"]
            - df["bollinger_lower_20"]
        )
        / df["bollinger_middle_20"]
    )

    return df


def create_candle_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["candle_range"] = (
        df["high"]
        - df["low"]
    )

    df["body_size"] = (
        df["close"]
        - df["open"]
    ).abs()

    df["upper_wick"] = (
        df["high"]
        - df[
            ["open", "close"]
        ].max(axis=1)
    )

    df["lower_wick"] = (
        df[
            ["open", "close"]
        ].min(axis=1)
        - df["low"]
    )

    df["range_pct"] = (
        df["candle_range"]
        / df["open"]
    )

    df["body_pct"] = (
        df["body_size"]
        / df["open"]
    )

    df["close_location"] = np.where(
        df["candle_range"] > 0,
        (
            df["close"]
            - df["low"]
        )
        / df["candle_range"],
        0.5,
    )

    return df


def create_volume_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["volume_change"] = (
        df["volume"]
        .pct_change()
    )

    for window in [
        10,
        30,
        60,
        240,
    ]:
        df[
            f"volume_sma_{window}"
        ] = (
            df["volume"]
            .rolling(window)
            .mean()
        )

    df["relative_volume_30"] = (
        df["volume"]
        / df["volume_sma_30"]
    )

    df["relative_volume_60"] = (
        df["volume"]
        / df["volume_sma_60"]
    )

    df["trade_count_change"] = (
        df["number_of_trades"]
        .pct_change()
    )

    df["trade_count_sma_30"] = (
        df["number_of_trades"]
        .rolling(30)
        .mean()
    )

    df["taker_buy_ratio"] = np.where(
        df["volume"] > 0,
        df["taker_buy_base_volume"]
        / df["volume"],
        0.5,
    )

    df["taker_buy_imbalance"] = (
        df["taker_buy_ratio"]
        - 0.5
    )

    return df


def create_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["hour"] = (
        df["open_time"].dt.hour
    )

    df["minute"] = (
        df["open_time"].dt.minute
    )

    df["day_of_week"] = (
        df["open_time"]
        .dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    minute_of_day = (
        df["hour"] * 60
        + df["minute"]
    )

    df["minute_of_day_sin"] = np.sin(
        2
        * np.pi
        * minute_of_day
        / 1440
    )

    df["minute_of_day_cos"] = np.cos(
        2
        * np.pi
        * minute_of_day
        / 1440
    )

    df["day_of_week_sin"] = np.sin(
        2
        * np.pi
        * df["day_of_week"]
        / 7
    )

    df["day_of_week_cos"] = np.cos(
        2
        * np.pi
        * df["day_of_week"]
        / 7
    )

    return df


def create_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    for lag in [
        1,
        2,
        3,
        5,
        10,
        20,
        30,
        60,
    ]:
        df[
            f"return_1m_lag_{lag}"
        ] = (
            df["return_1m"]
            .shift(lag)
        )

    for lag in [
        1,
        5,
        15,
        30,
        60,
    ]:
        df[
            f"volume_lag_{lag}"
        ] = (
            df["volume"]
            .shift(lag)
        )

    for lag in [
        1,
        3,
        5,
        15,
    ]:
        df[
            f"rsi_14_lag_{lag}"
        ] = (
            df["rsi_14"]
            .shift(lag)
        )

    for lag in [
        1,
        5,
        15,
    ]:
        df[
            f"volatility_30_lag_{lag}"
        ] = (
            df[
                "rolling_volatility_30"
            ]
            .shift(lag)
        )

    return df


def create_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["future_return_1m"] = (
        df["close"].shift(-1)
        / df["close"]
        - 1
    )

    return df


def main() -> None:
    print(
        "Loading BTCUSDT 1-minute data..."
    )

    df = load_data()

    print(
        f"Input rows: {len(df):,}"
    )

    df = create_return_features(df)
    df = create_trend_features(df)
    df = create_momentum_features(df)
    df = create_volatility_features(df)
    df = create_candle_features(df)
    df = create_volume_features(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_target(df)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    before = len(df)

    df = df.dropna().copy()

    removed = (
        before
        - len(df)
    )

    print(
        f"Rows removed for warm-up/NaN: "
        f"{removed:,}"
    )

    print(
        f"Final ML-ready rows: "
        f"{len(df):,}"
    )

    excluded = {
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
        "future_return_1m",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        "
1-minute future return statistics:"
    )

    print(
        df["future_return_1m"]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"
Saved:"
        f"
{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
