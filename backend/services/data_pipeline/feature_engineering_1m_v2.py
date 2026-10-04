from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from backend.services.data_pipeline.feature_engineering_1m import (
    create_candle_features,
    create_lag_features,
    create_momentum_features,
    create_return_features,
    create_target,
    create_time_features,
    create_trend_features,
    create_volatility_features,
    create_volume_features,
    load_data,
)


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_1m_features_v2.csv"
)


def create_price_action_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    # Prior rolling highs/lows are shifted by one candle so the
    # current candle is never used to define its own breakout level.
    for window in [
        10,
        20,
        60,
        120,
    ]:
        prior_high = (
            df["high"]
            .rolling(window)
            .max()
            .shift(1)
        )

        prior_low = (
            df["low"]
            .rolling(window)
            .min()
            .shift(1)
        )

        df[
            f"distance_to_{window}m_high"
        ] = (
            df["close"]
            / prior_high
            - 1
        )

        df[
            f"distance_to_{window}m_low"
        ] = (
            df["close"]
            / prior_low
            - 1
        )

        df[
            f"breakout_{window}m"
        ] = (
            df["close"]
            > prior_high
        ).astype(int)

        df[
            f"breakdown_{window}m"
        ] = (
            df["close"]
            < prior_low
        ).astype(int)

        rolling_range = (
            prior_high
            - prior_low
        )

        df[
            f"range_position_{window}m"
        ] = np.where(
            rolling_range > 0,
            (
                df["close"]
                - prior_low
            )
            / rolling_range,
            0.5,
        )

    # Simple market-structure signals.
    df["higher_high"] = (
        df["high"]
        > df["high"].shift(1)
    ).astype(int)

    df["higher_low"] = (
        df["low"]
        > df["low"].shift(1)
    ).astype(int)

    df["lower_high"] = (
        df["high"]
        < df["high"].shift(1)
    ).astype(int)

    df["lower_low"] = (
        df["low"]
        < df["low"].shift(1)
    ).astype(int)

    df["bullish_structure"] = (
        (
            df["higher_high"] == 1
        )
        & (
            df["higher_low"] == 1
        )
    ).astype(int)

    df["bearish_structure"] = (
        (
            df["lower_high"] == 1
        )
        & (
            df["lower_low"] == 1
        )
    ).astype(int)

    # Consecutive directional closes.
    up_close = (
        df["close"]
        > df["close"].shift(1)
    ).astype(int)

    down_close = (
        df["close"]
        < df["close"].shift(1)
    ).astype(int)

    df["up_closes_5"] = (
        up_close
        .rolling(5)
        .sum()
    )

    df["down_closes_5"] = (
        down_close
        .rolling(5)
        .sum()
    )

    return df


def create_vwap_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    price_volume = (
        typical_price
        * df["volume"]
    )

    for window in [
        20,
        60,
        240,
    ]:
        volume_sum = (
            df["volume"]
            .rolling(window)
            .sum()
        )

        pv_sum = (
            price_volume
            .rolling(window)
            .sum()
        )

        vwap = np.where(
            volume_sum > 0,
            pv_sum / volume_sum,
            np.nan,
        )

        df[
            f"vwap_{window}"
        ] = vwap

        df[
            f"distance_to_vwap_{window}"
        ] = (
            df["close"]
            / df[
                f"vwap_{window}"
            ]
            - 1
        )

    return df


def create_regime_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    # Trend relationships.
    df["ema_12_above_26"] = (
        df["ema_12"]
        > df["ema_26"]
    ).astype(int)

    df["ema_26_above_50"] = (
        df["ema_26"]
        > df["ema_50"]
    ).astype(int)

    df["ema_50_above_100"] = (
        df["ema_50"]
        > df["ema_100"]
    ).astype(int)

    df["ema_spread_12_26"] = (
        df["ema_12"]
        / df["ema_26"]
        - 1
    )

    df["ema_spread_26_50"] = (
        df["ema_26"]
        / df["ema_50"]
        - 1
    )

    # Volatility expansion / compression.
    short_vol = (
        df["rolling_volatility_10"]
    )

    medium_vol = (
        df["rolling_volatility_60"]
    )

    long_vol = (
        df["rolling_volatility_240"]
    )

    df["volatility_ratio_10_60"] = (
        short_vol
        / medium_vol
    )

    df["volatility_ratio_60_240"] = (
        medium_vol
        / long_vol
    )

    df["volatility_expansion"] = (
        short_vol
        > medium_vol
    ).astype(int)

    df["bollinger_compression"] = (
        df["bollinger_width_20"]
        / df[
            "bollinger_width_20"
        ]
        .rolling(120)
        .mean()
    )

    # Candle conviction.
    df["signed_body_pct"] = (
        (
            df["close"]
            - df["open"]
        )
        / df["open"]
    )

    df["body_to_range"] = np.where(
        df["candle_range"] > 0,
        df["body_size"]
        / df["candle_range"],
        0.0,
    )

    df["upper_wick_ratio"] = np.where(
        df["candle_range"] > 0,
        df["upper_wick"]
        / df["candle_range"],
        0.0,
    )

    df["lower_wick_ratio"] = np.where(
        df["candle_range"] > 0,
        df["lower_wick"]
        / df["candle_range"],
        0.0,
    )

    return df


def create_volume_confirmation_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["volume_breakout_30"] = (
        df["relative_volume_30"]
        > 1.5
    ).astype(int)

    df["volume_breakout_60"] = (
        df["relative_volume_60"]
        > 1.5
    ).astype(int)

    df["trade_intensity_30"] = (
        df["number_of_trades"]
        / df["trade_count_sma_30"]
    )

    df["buy_pressure_volume"] = (
        df["relative_volume_30"]
        * df["taker_buy_imbalance"]
    )

    df["breakout_with_volume_20"] = (
        (
            df["breakout_20m"] == 1
        )
        & (
            df["relative_volume_30"] > 1.2
        )
    ).astype(int)

    df["breakdown_with_volume_20"] = (
        (
            df["breakdown_20m"] == 1
        )
        & (
            df["relative_volume_30"] > 1.2
        )
    ).astype(int)

    return df


def main() -> None:
    print(
        "Loading BTCUSDT 1-minute data "
        "for chart-reading V2 features..."
    )

    df = load_data()

    print(
        f"Input rows: {len(df):,}"
    )

    # Preserve the complete V1 baseline feature set.
    df = create_return_features(df)
    df = create_trend_features(df)
    df = create_momentum_features(df)
    df = create_volatility_features(df)
    df = create_candle_features(df)
    df = create_volume_features(df)
    df = create_time_features(df)
    df = create_lag_features(df)

    # Add explicit price-action / chart-reading features.
    df = create_price_action_features(df)
    df = create_vwap_features(df)
    df = create_regime_features(df)
    df = create_volume_confirmation_features(df)

    # Keep the same one-minute target column for compatibility.
    # Multi-horizon tuning reconstructs 3m/5m/10m targets from close.
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
        f"Rows removed for warm-up/NaN: "
        f"{removed:,}"
    )

    print(
        f"Final ML-ready rows: "
        f"{len(df):,}"
    )

    print(
        f"V2 feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"Added chart-reading features: "
        f"{len(feature_columns) - 81}"
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
        f"\nSaved:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
