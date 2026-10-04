from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1h_context_features.csv"
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
        48,
        72,
    ]:
        df[
            f"return_{bars}h"
        ] = (
            df["close"]
            .pct_change(bars)
        )

    df["log_return_1h"] = np.log(
        df["close"]
        / df["close"].shift(1)
    )

    for bars in [
        3,
        6,
        12,
        24,
        48,
        72,
    ]:
        df[
            f"sma_{bars}h"
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
        48,
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

    for hours in [
        6,
        12,
        24,
        48,
        72,
    ]:
        df[
            f"close_to_sma_{hours}h"
        ] = (
            df["close"]
            / df[f"sma_{hours}h"]
            - 1
        )

    price_change = df["close"].diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    avg_gain = (
        gain.rolling(14).mean()
    )

    avg_loss = (
        loss.rolling(14).mean()
    )

    rs = avg_gain / avg_loss

    df["rsi_14"] = (
        100
        - (
            100
            / (
                1
                + rs
            )
        )
    )

    ema_fast = (
        df["close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema_slow = (
        df["close"]
        .ewm(
            span=26,
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
            span=9,
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
        48,
        72,
    ]:
        df[
            f"rolling_volatility_{bars}h"
        ] = (
            df["return_1h"]
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
        .rolling(20)
        .mean()
    )

    bollinger_std = (
        df["close"]
        .rolling(20)
        .std()
    )

    df["bollinger_middle_20h"] = (
        bollinger_middle
    )

    df["bollinger_upper_20h"] = (
        bollinger_middle
        + 2 * bollinger_std
    )

    df["bollinger_lower_20h"] = (
        bollinger_middle
        - 2 * bollinger_std
    )

    df["bollinger_width_20h"] = (
        (
            df["bollinger_upper_20h"]
            - df["bollinger_lower_20h"]
        )
        / df["bollinger_middle_20h"]
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
        48,
    ]:
        df[
            f"volume_sma_{bars}h"
        ] = (
            df["volume"]
            .rolling(bars)
            .mean()
        )

    df["relative_volume_6h"] = (
        df["volume"]
        / df["volume_sma_6h"]
    )

    df["relative_volume_24h"] = (
        df["volume"]
        / df["volume_sma_24h"]
    )

    df["trade_count_change"] = (
        df["number_of_trades"]
        .pct_change()
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

    df["day_of_week"] = (
        df["open_time"]
        .dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2
        * np.pi
        * df["hour"]
        / 24
    )

    df["hour_cos"] = np.cos(
        2
        * np.pi
        * df["hour"]
        / 24
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
        24,
    ]:
        df[
            f"return_1h_lag_{lag}"
        ] = (
            df["return_1h"]
            .shift(lag)
        )

    for lag in [
        1,
        3,
        6,
        12,
        24,
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

    return df


def main() -> None:
    print(
        "Loading freshly resampled "
        "BTCUSDT 1h data..."
    )

    df = load_data()

    print(
        f"Input rows: {len(df):,}"
    )

    df = create_features(df)

    df["future_return_1h"] = (
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
        "future_return_1h",
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
        f"Final context rows: "
        f"{len(df):,}"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Start: "
        f"{df['open_time'].iloc[0]}"
    )

    print(
        f"End:   "
        f"{df['open_time'].iloc[-1]}"
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
