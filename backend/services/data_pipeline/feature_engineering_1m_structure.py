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
    / "btcusdt_1m_features_structure.csv"
)


def create_market_structure_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    features = {}

    features["higher_high"] = (
        df["high"]
        > df["high"].shift(1)
    ).astype(int)

    features["higher_low"] = (
        df["low"]
        > df["low"].shift(1)
    ).astype(int)

    features["lower_high"] = (
        df["high"]
        < df["high"].shift(1)
    ).astype(int)

    features["lower_low"] = (
        df["low"]
        < df["low"].shift(1)
    ).astype(int)

    features["bullish_structure"] = (
        (
            features["higher_high"] == 1
        )
        & (
            features["higher_low"] == 1
        )
    ).astype(int)

    features["bearish_structure"] = (
        (
            features["lower_high"] == 1
        )
        & (
            features["lower_low"] == 1
        )
    ).astype(int)

    up_close = (
        df["close"]
        > df["close"].shift(1)
    ).astype(int)

    down_close = (
        df["close"]
        < df["close"].shift(1)
    ).astype(int)

    for window in [
        3,
        5,
        10,
    ]:
        features[
            f"up_closes_{window}"
        ] = (
            up_close
            .rolling(window)
            .sum()
        )

        features[
            f"down_closes_{window}"
        ] = (
            down_close
            .rolling(window)
            .sum()
        )

    structure_df = pd.DataFrame(
        features,
        index=df.index,
    )

    return pd.concat(
        [
            df,
            structure_df,
        ],
        axis=1,
    )


def main() -> None:
    print(
        "Loading BTCUSDT 1-minute data "
        "for market-structure ablation..."
    )

    df = load_data()

    print(
        f"Input rows: {len(df):,}"
    )

    # Exact V1 baseline.
    df = create_return_features(df)
    df = create_trend_features(df)
    df = create_momentum_features(df)
    df = create_volatility_features(df)
    df = create_candle_features(df)
    df = create_volume_features(df)
    df = create_time_features(df)
    df = create_lag_features(df)

    # Add only market-structure features.
    df = create_market_structure_features(
        df
    )

    df = create_target(df)

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
        "future_return_1m",
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

    print(
        f"Market-structure features added: "
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

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
