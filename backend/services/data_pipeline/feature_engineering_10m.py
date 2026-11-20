from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_10m_features.csv"
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

    return (
        df.sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    for bars in [
        1,
        2,
        3,
        6,
        12,
        24,
    ]:
        df[
            f"return_{bars * 10}m"
        ] = (
            df["close"]
            .pct_change(bars)
        )

    df["log_return_10m"] = np.log(
        df["close"]
        / df["close"].shift(1)
    )

    for bars in [
        3,
        6,
        12,
        24,
    ]:
        df[
            f"sma_{bars * 10}m"
        ] = (
            df["close"]
            .rolling(bars)
            .mean()
        )

    for span in [
        3,
        6,
        12,
        24,
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

    df["close_to_sma_30m"] = (
        df["close"]
        / df["sma_30m"]
        - 1
    )

    df["close_to_sma_60m"] = (
        df["close"]
        / df["sma_60m"]
        - 1
    )

    df["close_to_sma_240m"] = (
        df["close"]
        / df["sma_240m"]
        - 1
    )

    price_change = df["close"].diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = (
        gain.rolling(14).mean()
    )

    average_loss = (
        loss.rolling(14).mean()
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

    ema_fast = (
        df["close"]
        .ewm(
            span=3,
            adjust=False,
        )
        .mean()
    )

    ema_slow = (
        df["close"]
        .ewm(
            span=6,
            adjust=False,
        )
        .mean()
    )

    df["macd"] = (
        ema_fast
        - ema_slow
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=3,
            adjust=False,
        )
        .mean()
    )

    df["macd_histogram"] = (
        df["macd"]
        - df["macd_signal"]
    )

    for bars in [
        3,
        6,
        12,
        24,
    ]:
        df[
            f"rolling_volatility_{bars * 10}m"
        ] = (
            df["return_10m"]
            .rolling(bars)
            .std()
        )

    previous_close = (
        df["close"].shift(1)
    )

    true_range = pd.concat(
        [
            df["high"] - df["low"],
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
        .rolling(6)
        .mean()
    )

    bollinger_std = (
        df["close"]
        .rolling(6)
        .std()
    )

    df["bollinger_middle_60m"] = (
        bollinger_middle
    )

    df["bollinger_upper_60m"] = (
        bollinger_middle
        + 2 * bollinger_std
    )

    df["bollinger_lower_60m"] = (
        bollinger_middle
        - 2 * bollinger_std
    )

    df["bollinger_width_60m"] = (
        (
            df["bollinger_upper_60m"]
            - df["bollinger_lower_60m"]
        )
        / df["bollinger_middle_60m"]
    )

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

    df["volume_change"] = (
        df["volume"]
        .pct_change()
    )

    for bars in [
        3,
        6,
        12,
        24,
    ]:
        df[
            f"volume_sma_{bars * 10}m"
        ] = (
            df["volume"]
            .rolling(bars)
            .mean()
        )

    df["relative_volume_30m"] = (
        df["volume"]
        / df["volume_sma_30m"]
    )

    df["relative_volume_60m"] = (
        df["volume"]
        / df["volume_sma_60m"]
    )

    df["trade_count_change"] = (
        df["number_of_trades"]
        .pct_change()
    )

    df["trade_count_sma_30m"] = (
        df["number_of_trades"]
        .rolling(3)
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

    for lag in [
        1,
        2,
        3,
        6,
        12,
    ]:
        df[
            f"return_10m_lag_{lag}"
        ] = (
            df["return_10m"]
            .shift(lag)
        )

    for lag in [
        1,
        3,
        6,
        12,
    ]:
        df[
            f"volume_lag_{lag}"
        ] = (
            df["volume"]
            .shift(lag)
        )

    for lag in [
        1,
        2,
        3,
        6,
    ]:
        df[
            f"rsi_14_lag_{lag}"
        ] = (
            df["rsi_14"]
            .shift(lag)
        )

    for lag in [
        1,
        3,
        6,
    ]:
        df[
            f"volatility_30m_lag_{lag}"
        ] = (
            df[
                "rolling_volatility_30m"
            ]
            .shift(lag)
        )

    return df


def main() -> None:
    print(
        "Loading BTCUSDT 10-minute data..."
    )

    df = load_data()

    print(
        f"Input rows: {len(df):,}"
    )

    df = create_features(df)

    df["future_return_10m"] = (
        df["close"].shift(-1)
        / df["close"]
        - 1
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    before = len(df)

    df = df.dropna().copy()

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
        "future_return_10m",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    print(
        f"Rows removed for warm-up/NaN: "
        f"{before - len(df):,}"
    )

    print(
        f"Final ML-ready rows: "
        f"{len(df):,}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print()
    print(
        "10-minute future return statistics:"
    )

    print(
        df["future_return_10m"]
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

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
